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
 * This container class supports two basic mutable operations, both of
 * which can be performed **concurrently** and with nearly **constant
 * time** (really, it is linear with very low slope):
 *
 * - *push(item)*: append an element at the end, returns its index
 * - *remove(idx)*: deletes the element given its index
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
 * New elements can only be `push`ed to the end.
 * They can not be inserted at random position.
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

        void reset(T const & value)
        {
            storage.value = value;            
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

    struct Chunk
    {
        /* `item_count` starts with 1 to keep the chunk at least until
         * its full capacity is used.  This initial offset is
         * compensated by not increasing the count when the last
         * element is inserted.
         */
        std::atomic< uint16_t > item_count;
        Item * items;

        Chunk()
            : items(nullptr)
            , item_count(1)
        {}

        Chunk( Chunk && other )
            :items(other.items)
            ,item_count(other.item_count)
        {}

        Chunk & operator=(Chunk const & other)
        {
            items = other.items;
            item_count = other.item_count.load();
            return *this;
        }

        void allocate( memory::Allocator< Item > & item_alloc, size_t chunk_size )
        {
            items = item_alloc.allocate( chunk_size );
            // memory is cleared by allocator
            //memset(items, 0, sizeof(Item) * chunk_size);
        }

        void deallocate( memory::Allocator< Item > & item_alloc )
        {
            Item * it = items;
            items = nullptr;
            if( it )
                item_alloc.deallocate( it );
        }

        void erase_item( memory::Allocator< Item > & item_alloc )
        {
            if( item_count.fetch_sub(1) == 1 )
            {
                // last element was deleted from this chunk,
                // deallocate memory now
                deallocate( item_alloc );
            }
        }
    };

    struct ItemAccess
    {
    private:
        friend class ChunkedList;

        Chunk & chunk;
        memory::Allocator< Item > item_alloc;

        bool has_item;
        unsigned chunk_off;

        ItemAccess( Chunk & chunk, memory::Allocator< Item > const & item_alloc, unsigned chunk_off )
            : chunk(chunk), item_alloc(item_alloc), chunk_off(chunk_off)
        {
            if( item().refcount > 0 )
            {
                item().refcount++;
                has_item = true;
            }
            else
                has_item = false;
        }

    public:
        ItemAccess( ItemAccess const & other )
            : ItemAccess( other.chunk, other.item_alloc, other.chunk_off )
        {
        }

        ~ItemAccess()
        {
            if( has_item )
                if( item().refcount.fetch_sub(1) == 1 )
                    chunk.erase_item( item_alloc );
        }

        bool is_some() const
        {
            return has_item;
        }

        Item & item() const
        {
            return chunk.items[chunk_off];
        }

        T& operator* () const
        {
            assert( is_some() );
            return item().storage.value;
        }
    };

private:

    std::atomic< uint16_t > next_superchunk_idx;
    std::atomic< uint16_t > superchunk_size;
    std::atomic< uint16_t > superchunk_capacity;
    Chunk * superchunk;

    std::atomic< uint32_t > next_item_id;
    std::atomic< uint32_t > size_;

    memory::Allocator< Item >  item_alloc;
    memory::Allocator< Chunk > chunk_alloc;

    // 
    SpinLock m;

    /* Allocates a new chunk and inserts a reference to it in the
     * superchunk.
     * If necessary, the superchunk is resized.
     * Mutex `m` is required to be locked whenever this function is called.
     */
    void init_chunk( )
    {
        unsigned chunk_idx = next_superchunk_idx.fetch_add(1);

        while( chunk_idx >= superchunk_capacity )
            resize_superchunk( superchunk_capacity + 8 );

        superchunk[ chunk_idx ].allocate( item_alloc, chunk_size );

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
            new_superchunk[i] = std::move(superchunk[i]);

        for( int i = superchunk_capacity; i < new_superchunk_capacity; ++i )
        {
            new_superchunk[i].item_count = 0;
            new_superchunk[i].items = nullptr;
        }

        auto old_superchunk = superchunk;
        superchunk = new_superchunk;

        chunk_alloc.deallocate( old_superchunk, superchunk_capacity );
        superchunk_capacity = new_superchunk_capacity;
    }

    public:
        ~ChunkedList()
        {
            for( unsigned i = 0; i < superchunk_size; ++i )
                superchunk[i].deallocate( item_alloc );

            chunk_alloc.deallocate( superchunk, superchunk_capacity );
        }

        ChunkedList( size_t chunk_size = 16 )
            : ChunkedList(
                memory::Allocator< Item >(),
                memory::Allocator< Chunk >(),
                chunk_size
            )
        {}

        ChunkedList(
            memory::Allocator< Item > item_alloc,
            memory::Allocator< Chunk > chunk_alloc,
            size_t chunk_size = 16
        )
            : item_alloc( item_alloc )
            , chunk_alloc( chunk_alloc )
            , chunk_size( chunk_size )
            , next_item_id( 0 )
            , size_( 0 )
            , superchunk_size( 0 )
            , superchunk_capacity( 8 )
        {
            superchunk = chunk_alloc.allocate( superchunk_capacity );
            for( int i = 0; i < superchunk_capacity; ++i )
                new (&superchunk[i]) Chunk();
        }

        ChunkedList( ChunkedList && other ) = default;

        ChunkedList( ChunkedList const& other )
            : ChunkedList( other.item_alloc, other.chunk_alloc, other.chunk_size )
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

            if( superchunk[ chunk_idx ].items )
                return ItemAccess( superchunk[ chunk_idx ], item_alloc, chunk_off );
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
                //std::unique_lock< SpinLock > l(m);

                while( chunk_idx >= superchunk_capacity
                       || superchunk[ chunk_idx ].items == nullptr )
                    init_chunk();

                size_ ++;

                // except for the last item, every insertion increases
                // the `item_count` of current chunk
                if( chunk_off+1 < chunk_size )
                    superchunk[ chunk_idx ].item_count ++;

                // insert element
                superchunk[ chunk_idx ].items[ chunk_off ].reset(item);
            }

            return idx;
        }

        void remove( unsigned idx )
        {
            assert( idx < size() );
            unsigned chunk_idx = idx / chunk_size;
            unsigned chunk_off = idx % chunk_size;

            if( get(idx).item().refcount.fetch_add(1) == 1 )
            {
                // no further references to this item exist,
                // delete it from superchunk
                superchunk[ chunk_idx ].erase_item( item_alloc );
            }
        }

        void erase( T item )
        {
            for( auto it = begin(); it != end(); ++it)
                if( *it == item )
                    remove( it.idx );
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

