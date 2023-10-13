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
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <redGrapes/util/allocator.hpp>
#include <redGrapes/util/bump_alloc_chunk.hpp>
#include <redGrapes/util/spinlock.hpp>
#include <redGrapes/util/chunklist.hpp>
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
 */



/*
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
template <
    typename T,
    template <typename> class Allocator = memory::Allocator
>
struct ChunkedList
{
    size_t const chunk_size;

    using chunk_offset_t = uint16_t;
    using refcount_t = uint16_t;

    struct Item
    {
        struct TrivialInit_t{};
        union ItemStorage
        {
            char dummy;
            T value;

            ItemStorage( TrivialInit_t ) noexcept
                : dummy()
            {}

            template < typename... Args >
            ItemStorage( Args&&... args )
                : value(std::forward<Args>(args)...)
            {}

            ~ItemStorage() {}
        };

        /* in case this item is deleted, `iter_offset` gives an
         * offset by which we can safely jump to find the next
         * existing item.
         * iter_offset = 0 means this item exists
         * iter_offset = 1 means previous item exists
         * ...
         */
        std::atomic< chunk_offset_t > iter_offset;

        /* counts the number of iterators pointing
         * at this item currently
         */
        std::atomic< refcount_t > refcount;

        /* actual data
         */
        ItemStorage storage;

        Item()
            : iter_offset( 0 )
            , refcount( 0 )
            , storage( TrivialInit_t{} )
        {}

        T & operator=(T const & value)
        {
            if( refcount.fetch_add(1) == 0 )
            {
                storage.value = value;
                iter_offset = 0;
                return storage.value;
            }
            else
                throw std::runtime_error("assign item which is in use");
        }

        /* decrement refcount and in case this 
         * was the last reference, deconstruct the element
         */
        void remove()
        {
            if( refcount.fetch_add(1) == 1 )
                storage.value.~T();
        }
    };

    struct Chunk
    {
        /* counts the number of alive elements in this chunk.
         * Whenever `item_count` reaches zero, the chunk will be deleted.
         * `item_count` starts with 1 to keep the chunk at least until
         * its full capacity is used.  This initial offset is
         * compensated by not increasing the count when the last
         * element is inserted.
         */
        std::atomic< chunk_offset_t > item_count{ 1 };

        /* lowest index with free slot that can
         * be used to add a new element
         */
        std::atomic< chunk_offset_t > next_idx{ 0 };

        /* highest index with fully initialized item
         * where the iterator can start
         */
        std::atomic< chunk_offset_t > last_idx{ 0 };

        Chunk( size_t chunk_size )
        {
            for( unsigned i= 0; i < chunk_size; ++i )
                new ( &items()[i] ) Item();
        }

        Item * items()
        {
            return (Item*)( (uintptr_t)this + sizeof(Chunk) );
        }
    };

    template < bool is_const >
    struct ItemAccess
    {
    private:
        friend class ChunkedList;

        chunk_offset_t chunk_size;
        chunk_offset_t chunk_off;
        bool has_item;
        typename memory::ChunkList< Chunk, Allocator >::MutBackwardIterator chunk;

    protected:
        bool is_valid_idx() const
        {
            return ( (bool)chunk ) && ( chunk_off < chunk_size );
        }

        bool try_acquire()
        {
            if( is_valid_idx() )
            {
                refcount_t old_refcount = item().refcount.fetch_add(1);
                if( old_refcount >= 1 )
                {
                    chunk_offset_t off = item().iter_offset.load();
                    if( off == 0 )
                        has_item = true;
                }
            }
            return has_item;
        }

        void release()
        {
            if( has_item )
                item().remove();

            has_item = false;
        }

        void acquire_next_item()
        {
            while( is_valid_idx() )
            {
                uint16_t step = item().iter_offset;

                if( step == 0 )
                {
                    if( try_acquire() )
                        return;
                }
                else
                {
                    if( step <= chunk_off )
                        chunk_off -= step;
                    else
                    {
                        ++chunk;
                        chunk_off = chunk_size - 1;
                    }
                }
            }

            chunk_off = std::numeric_limits< chunk_offset_t >::max();
        }

        ItemAccess( size_t chunk_size,
                    typename memory::ChunkList< Chunk, Allocator >::MutBackwardIterator chunk,
                    unsigned chunk_off )
            : has_item(false), chunk_size(chunk_size), chunk(chunk), chunk_off(chunk_off)
        {
            acquire_next_item();
        }

        Item & item() const
        {
            assert( is_valid_idx() );
            return chunk->items()[chunk_off];
        }

    public:
        ItemAccess( ItemAccess const & other )
            : ItemAccess( other.chunk_size, other.chunk, other.chunk_off )
        {
        }

        ~ItemAccess()
        {
            release();
        }

        bool is_valid() const
        {
            return has_item;
        }

        typename std::conditional<
            is_const,
            T const *,
            T *
        >::type
        operator->() const
        {
            assert( is_valid() );
            return &item().storage.value;
        }

        typename std::conditional<
            is_const,
            T const &,
            T &
        >::type
        operator* () const
        {
            assert( is_valid() );
            return item().storage.value;
        }
    };

    template < bool is_const >
    struct BackwardIterator : ItemAccess< is_const >
    {
        BackwardIterator(
            size_t chunk_size,
            typename memory::ChunkList< Chunk, Allocator >::MutBackwardIterator chunk,
            unsigned chunk_off
        )
            : ItemAccess< is_const >( chunk_size, chunk, chunk_off )
        {
        }

        bool operator!=(BackwardIterator< is_const > const& other)
        {
            if( !this->is_valid_idx() && !other.is_valid_idx() )
                return false;
                
            return this->chunk != other.chunk
                || this->chunk_off != other.chunk_off;
        }

        BackwardIterator< is_const > & operator=( BackwardIterator< is_const > const & other )
        {
            this->release();
            this->chunk_off = other.chunk_off;
            this->chunk = other.chunk;
            this->chunk_size = other.chunk_size;
            this->try_acquire();
            return *this;
        }

        BackwardIterator & operator++()
        {
            this->release();

            if( this->chunk_off > 0 )
                -- this->chunk_off;
            else
            {
                ++ this->chunk;
                this->chunk_off = this->chunk_size - 1;
            }

            this->acquire_next_item();
            return *this;
        }
    };

    using ConstBackwardIterator = BackwardIterator< true >;
    using MutBackwardIterator = BackwardIterator< false >;

private:
    memory::ChunkList< Chunk, Allocator > chunks;

public:
    ChunkedList( size_t chunk_size = 16 )
        : ChunkedList(
            Allocator< uint8_t >(),
            chunk_size
        )
    {}

    ChunkedList(
        Allocator< uint8_t > alloc,
        size_t chunk_size = 64
    )
        : chunk_size( chunk_size )
        , chunks( alloc, sizeof(Chunk) + sizeof(Item)*chunk_size )
    {
        assert( chunk_size < std::numeric_limits< chunk_offset_t >::max() );
    }

    ChunkedList( ChunkedList && other ) = default;

    ChunkedList( Allocator< uint8_t > alloc, ChunkedList const & other )
        : ChunkedList( alloc, other.chunk_size )
    {
        spdlog::error("copy construct ChunkedList!!");
    }

    MutBackwardIterator push( T const& item )
    {
        TRACE_EVENT("ChunkedList", "push");

        while( true )
        {
            auto chunk = chunks.rbegin();
            if( chunk != chunks.rend() )
            {
                unsigned chunk_off = chunk->next_idx.fetch_add(1);

                if( chunk_off < chunk_size )
                {
                    chunk->item_count ++;
                    chunk->items()[ chunk_off ] = item;
                    chunk->last_idx ++;
                    return MutBackwardIterator( chunk_size, chunk, chunk_off );
                }
            }

            chunks.add_chunk( chunk_size );
        }
    }

    void remove( MutBackwardIterator & pos )
    {
        /* first, set iter_offset, so that any iterator
         * will skip this element from now on
         */

        if( pos.is_valid_idx() )
        {
            // first elements just goes back one step to reach last element of previous chunk
            if( pos.chunk_off == 0 )
                pos.chunk->items()[ pos.chunk_off ].iter_offset = 1;

            // if we have a predecessor in this chunk, reuse their offset
            else
                pos.chunk->items()[ pos.chunk_off ].iter_offset =
                    pos.chunk->items()[ pos.chunk_off - 1 ].iter_offset + 1;
        }
        else
            throw std::runtime_error("remove invalid position");

        /* decrement refcount once so the item will be deconstructed
         * eventually, when all iterators drop their references
         */
        pos.item().remove();

        /* in case all items of this chunk are deleted,
         * delete the chunk too
         */
        if( pos.chunk->item_count.fetch_sub(1) == 1 )
            chunks.erase( pos.chunk );
    }

    void erase( T item )
    {
        for( auto it = rbegin(); it != rend(); ++it )
            if( *it == item )
                remove( it );
    }

    MutBackwardIterator rbegin() const
    {
        auto c = chunks.rbegin();
        return MutBackwardIterator(
            chunk_size,
            c,
            ( c != chunks.rend() ) ?
                c->last_idx.load()-1
                : std::numeric_limits< chunk_offset_t >::max()
        );
    }

    MutBackwardIterator rend() const
    {
        return MutBackwardIterator(
            chunk_size,
            chunks.rend(),
            std::numeric_limits< chunk_offset_t >::max()
        );
    }

    ConstBackwardIterator crbegin() const
    {
        auto c = chunks.rbegin();
        return ConstBackwardIterator(
            chunk_size,
            c,
            ( c != chunks.rend() ) ?
                c->last_idx.load()-1
                : std::numeric_limits< chunk_offset_t >::max()
        );
    }

    ConstBackwardIterator crend() const
    {
        return ConstBackwardIterator(
            chunk_size,
            chunks.rend(),
            std::numeric_limits< chunk_offset_t >::max()
        );
    }
};

} // namespace redGrapes

