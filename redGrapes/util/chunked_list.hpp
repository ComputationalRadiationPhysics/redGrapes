/* Copyright 2022-2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/util/chunked_list.hpp
 */

#pragma once

#include <cassert>
#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <redGrapes/util/allocator.hpp>
#include <redGrapes/util/spinlock.hpp>
#include <redGrapes/util/trace.hpp>
#include <spdlog/spdlog.h>

namespace redGrapes
{

/*!
 * TODO: Name?
 * ALRARAC - Allocation & Lock Reducing Append/Remove Array Container
 * LARAC- Lock- & Allocation-Reducing Array Container
 *
 * This container class supports two basic mutable operations,
 * both of which can be performed **concurrently** and with nearly **constant time**:
 *    - *push()*: append an element at the end, returns its index
 *    - *remove()*: deletes the element given its index
 *
 * It is implemented as two-Level pointer tree where
 * the first level is a dynamicly sized array, called *SuperChunk*.
 * This *SuperChunk* references a growing number of *Chunks*,
 * a staticly-sized array each.
 *
 *    [ ChkPtr_1, ChkPtr_2, .., ChkPtr_chunk_count ]
 *        |           |_________________     |_____
 *       V                              V          V
 *    [ x_1, x_2, .., x_chunk_size ] [ ... ] .. [ ... ]
 *
 * New elements can only be `push`ed to the end,
 * but can not be inserted at random position.
 * Depending on `chunk_size` , adding new elements
 * is performed in constant time and without allocations,
 * as long as the chunk still has capacity.
 * Only one of chunk_size many calls of push() require
 * memory allocation.
 *
 * Elements are removed in constant time and
 * removed elements are skipped by the iterators,
 * however their memory is still occupied
 * until all elements of the chunk are removed.
 *
 * Iteration can be started at a specific index and
 * proceed in forward direction (++) aswell as reversed (--).
 *
 * Indices returned by push() can be used with begin_from() / remove().
 * These indices are static, i.e. they stay valid after other remove() operations.
 *
 *
 * ## Example Usecases:
 *
 *  - **Resource User List** (exist per resource) is used concurrently with:
 *     - push() from mostly one but potentially many task-creator-threads through emplace_task(),
 *     - reversed iteration and remove() from Worker threads
 *       from task.init_graph() and task.remove_from_resources().
 *
 *  - **Event Follower List** (exists per event):
 *     - push() concurrently by multiple Workers initializing new task dependenies,
 *     - remove() concurrently my mutliple Workers through update_graph().
 *
 *  - **Access List** (exists per task):
 *     - push() from only one single thread that is initializing the task
 *     and after that finished,
 *     - remove() from only one single thread.
 *     - concurrently to the first two, all Worker threads may iterate read-only.
 */
template < typename T >
struct ChunkedList
{
private:
    size_t const chunk_size;

    struct Item
    {
        struct TrivialInit_t{};
        union ItemStorage
        {
            unsigned char dummy;
            T value;

            ItemStorage( TrivialInit_t ) noexcept
                : dummy()
            {}

            template < typename... Args >
            ItemStorage( Args&&... args )
                : value(std::forward<Args>(args)...)
            {}

            ~ItemStorage(){}
        };

        std::atomic< uint8_t > refcount;
        ItemStorage storage;

        Item()
            : refcount( 0 )
            , storage( TrivialInit_t{} )
        {
        }

        T & operator=(T const & value)
        {
            if( refcount.fetch_add(1) == 0 )
            {
                storage.value = value;
                return storage.value;
            }
            else
                throw std::runtime_error("assign item which is in use");
        }
    };

    struct ItemAccess
    {
    private:
        friend class ChunkedList;

        bool has_item;
        Item & item;

        ItemAccess( Item & item )
            : item(item)
        {
            if( item.refcount > 0 )
            {
                item.refcount++;
                has_item = true;
            }
            else
                has_item = false;
        }

    public:
        ItemAccess( ItemAccess const & other )
            : ItemAccess( other.item )
        {
        }

        ~ItemAccess()
        {
            if( has_item )
                item.refcount--;
        }

        bool is_some() const
        {
            return has_item;
        }

        T& operator* () const
        {
            assert( is_some() );
            return item.storage.value;
        }
    };
    
    using Chunk = Item *;

    std::atomic< uint16_t > superchunk_size;
    std::atomic< uint16_t > superchunk_capacity;
    Chunk * superchunk;

    std::atomic< uint32_t > next_item_id;
    std::atomic< uint32_t > size_;

    memory::Allocator< Item >  item_alloc;
    memory::Allocator< Chunk > chunk_alloc;

    // 
    SpinLock m;

    // m is locked whenever this function is called
    void init_chunk( )
    {
        Item * new_chunk = item_alloc.allocate(chunk_size);
        for( int i = 0; i < chunk_size; ++i )
            new (&new_chunk[i]) Item;

        unsigned chunk_idx = superchunk_size;

        if( chunk_idx >= superchunk_capacity )
            resize_superchunk( superchunk_capacity + 8 );

        superchunk[ chunk_idx ] = new_chunk;

        superchunk_size ++;
    }

    void resize_superchunk( size_t new_superchunk_capacity )
    {
        assert( new_superchunk_capacity > superchunk_capacity );

        SPDLOG_TRACE("resize superchunk to {} chunks", new_superchunk_capacity);

        // TODO: Optimization
        //  In the case that the allocated memory region
        //  for the new superchunk directly follows the old one,
        //  we dont need to copy the elements, just 'merge' the two
        //  and set the superchunk_capacity appropriately.
        // Question: Where to store old pointer to free it later?
        // Trick: ChunkAllocator only cares about count of free()'s, not the pointer

        Chunk * new_superchunk = chunk_alloc.allocate( new_superchunk_capacity );

        for( int i = 0; i < superchunk_capacity; ++i )
            new_superchunk[i] = superchunk[i];

        for( int i = superchunk_capacity; i < new_superchunk_capacity; ++i )
            new_superchunk[i] = nullptr;

        auto old_superchunk = superchunk;
        superchunk = new_superchunk;

        chunk_alloc.deallocate( old_superchunk, superchunk_capacity );
        superchunk_capacity = new_superchunk_capacity;
    }

    public:
        ~ChunkedList()
        {
            for( unsigned i = 0; i < superchunk_size; ++i )
                if( superchunk[i] )
                    item_alloc.deallocate( superchunk[i], chunk_size );

            chunk_alloc.deallocate( superchunk, superchunk_capacity );
        }

        ChunkedList( size_t chunk_size = 256 )
            : chunk_size( chunk_size )
            , next_item_id( 0 )
            , size_( 0 )
            , superchunk_size( 0 )
            , superchunk_capacity( 16 )
        {
            superchunk = chunk_alloc.allocate( superchunk_capacity );
            for( int i = 0; i < superchunk_capacity; ++i )
                superchunk[i] = nullptr;
        }

        ChunkedList( ChunkedList && other ) = default;

        ChunkedList( ChunkedList const& other )
            : ChunkedList()
        {
            SPDLOG_TRACE("copy construct ChunkedList!!");
            reserve( other.size() );
            for( auto& e : other )
                push(e);
        }

        unsigned size() const noexcept
        {
            return size_;
        }

        unsigned capacity() const noexcept
        {
            return superchunk_size * chunk_size;
        }

        unsigned free_capacity() const noexcept
        {
            return capacity() - next_item_id;
        }

        void reserve( size_t required_size )
        {
            if( required_size > free_capacity() ) {
                reserve_chunks(
                    1 +
                    ((required_size - free_capacity() - 1) / chunk_size)
                );
            }
        }

        void reserve_chunks( size_t n )
        {
            //spdlog::info("reserve {} chunks", n);

            std::unique_lock< SpinLock > l(m);

            if( superchunk_size + n > superchunk_capacity )
                resize_superchunk( superchunk_size + n );

            assert( superchunk_size + n <= superchunk_capacity );

            for( size_t i = 0; i < n; ++i )
                init_chunk();
        }

        ItemAccess get( unsigned idx ) const
        {
            unsigned chunk_idx = idx / chunk_size;
            unsigned chunk_off = idx % chunk_size;

            assert( chunk_idx < superchunk_size );

            if( superchunk[ chunk_idx ] )
                return ItemAccess( superchunk[ chunk_idx ][ chunk_off ] );
            else
                throw std::runtime_error(fmt::format("invalid index: missing chunk {}", chunk_idx));
        }

        unsigned push( T const& item )
        {
            TRACE_EVENT("ChunkedList", "push");
            unsigned idx = next_item_id.fetch_add(1);
            unsigned chunk_idx = idx / chunk_size;
            unsigned chunk_off = idx % chunk_size;

            {
                // TODO: reduce this critical section
                std::unique_lock< SpinLock > l(m);

                while( chunk_idx >= superchunk_capacity
                       || superchunk[ chunk_idx ] == nullptr )
                    init_chunk();

                superchunk[ chunk_idx ][ chunk_off ] = item;
                size_ ++;
            }

            return idx;
        }

        void remove(unsigned idx)
        {
            assert( idx < size() );
            get(idx).item.refcount--;
        }

        void erase( T item )
        {
            for( unsigned i = 0; i < size(); ++i )
            {
                ItemAccess x = get(i);
                if( x.is_some() )
                {
                    if( *x == item )
                    {
                        remove( i );
                        //return;
                    }
                }
            }
        }

        // TODO: can we have one iterator class for const and non-const ?

        /* ITERATOR */
        /************/
        struct ConstIterator
        {
            ChunkedList< T > const & c;
            int idx;

            bool operator!=(ConstIterator const& other)
            {
                return this->idx != other.idx;
            }

            ConstIterator& operator--()
            {
                do {
                    --idx;
                    if( idx < 0 )
                        return *this;
                }
                while( !is_some() );

                return *this;
            }

            ConstIterator& operator++()
            {
                do {
                    ++idx;
                    if( idx >= c.size() )
                    {
                        idx = -1;
                        return *this;
                    }
                }
                while( !is_some() );

                return *this;
            }

            bool is_some() const
            {
                if( idx < c.size() && idx >= 0 )
                    return c.get(idx).is_some();
                else
                    return false;
            }

            bool is_empty() const
            {
                return c.size() == 0;
            }

            T const & operator*() const
            {
                assert( is_some() );
                return *c.get(idx);
            }            
        };

        struct Iterator
        {
            ChunkedList< T > & c;
            int idx;

            bool operator!=(Iterator const& other)
            {
                return this->idx != other.idx;
            }

            Iterator& operator--()
            {
                do {
                    --idx;
                    if( idx < 0 )
                        return *this;
                }
                while( !is_some() );

                return *this;
            }

            Iterator& operator++()
            {
                do {
                    ++idx;
                    if( idx >= c.size() )
                    {
                        idx = -1;
                        return *this;
                    }
                }
                while( !is_some() );

                return *this;
            }

            bool is_some() const
            {
                if( idx < c.size() && idx >= 0 )
                    return c.get(idx).is_some();
                else
                    return false;
            }

            T & operator*() const
            {
                assert( is_some() );
                return *c.get(idx);
            }
        };

        auto begin_from(unsigned idx)
        {
            if( size() == 0 )
                return end();

            auto it = Iterator{ *this, (int)idx };

            while( ! it.is_some() && it != end() )
                ++it;

            return it;
        }

        auto begin_from(unsigned idx) const
        {
            if( size() == 0 )
                return end();

            auto it = ConstIterator{ *this, (int)idx };

            while( ! it.is_some() && it != end() )
                ++it;

            return it;
        }
    
        auto begin_from_rev(unsigned idx)
        {
            if( size() == 0 )
                return end();

            auto it = Iterator{ *this, (int)idx };

            while( ! it.is_some() && it != end() )
                --it;

            return it;
        }

        auto begin_from_rev(unsigned idx) const
        {
            if( size() == 0 )
                return end();

            auto it = ConstIterator{ *this, (int)idx };

            while( ! it.is_some() && it != end() )
                --it;

            return it;
        }
    
        auto begin() const
        {
            return begin_from(0);
        }

        auto begin()
        {
            return begin_from(0);
        }

        auto end()
        {
            return Iterator{ *this, -1 };
        }

        auto end() const
        {
            return ConstIterator{ *this, -1 };
        }

    };

} // namespace redGrapes

