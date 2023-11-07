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
#include <redGrapes/util/trace.hpp>
#include <redGrapes/memory/allocator.hpp>
#include <redGrapes/memory/bump_allocator.hpp>
#include <redGrapes/util/spinlock.hpp>
#include <redGrapes/util/atomic_list.hpp>
#include <spdlog/spdlog.h>

namespace redGrapes
{

/*!
 * This container class supports two basic mutable, iterator-stable operations,
 * both of which can be performed **concurrently** an in nearly **constant time**:
 *
 * - *push(item)*: append an element at the end, returns its index
 * - *remove(idx)*: deletes the element given its index
 *
 * It is implemented as atomic linked list of chunks,
 * which are fixed size arrays of elements.
 *
 * New elements can only be `push`ed to the end.
 * They can not be inserted at random position.
 * Depending on `chunk_size` , adding new elements
 * is performed in constant time and without allocations,
 * as long as the chunk still has capacity.
 * Only one of chunk_size many calls of push() require
 * memory allocation.
 *
 * Elements are removed in constant time.
 * Removed elements are skipped by the iterators,
 * however their memory is still occupied
 * until all elements of the chunk are removed.
 * The instances of items are kept alive until all
 * iterators referencing that item have released the
 * ownership. Then the element-destructor is called.
 *
 * Iteration can begin at a specific position that was
 * returned by `push`.
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

        ~Item()
        {
            if( refcount.fetch_sub(1) == 1 )
                storage.value.~T();
        }

        /* initialize value of this item.
         * only intended for new elements,
         * re-assigning is not allowed.
         */
        T & operator=(T const & value)
        {
            auto old_refcount = refcount.fetch_add(1);
            assert( old_refcount == 0 );

            storage.value = value;
            iter_offset = 0;
            return storage.value;
        }

        /* decrement refcount and in case this 
         * was the last reference, deconstruct the element
         */
        void remove()
        {
            refcount_t old_refcount = refcount.fetch_sub(1);

            if( old_refcount == 1 )
                storage.value.~T();

            if( old_refcount == 0 )
                throw std::runtime_error("remove inexistent item");
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

        Chunk( uintptr_t lower_limit, uintptr_t upper_limit )
        {
            size_t n = (upper_limit - lower_limit) / sizeof(Item);
            for( unsigned i= 0; i < n; ++i )
                new ( &items()[i] ) Item();
        }

        ~Chunk()
        {
            for( unsigned i = 0; i < last_idx; ++i )
                items()[i].~Item();
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
        typename memory::AtomicList< Chunk, Allocator >::MutBackwardIterator chunk;

    protected:
        /*!
         * checks whether the iterator points to an existing storage location.
         * This storage location can be used, free or deleted.
         * Only by `rend()`, and if the container is empty also `rbegin()` shall
         * return an iterator with invalid idx.
         */
        bool is_valid_idx() const
        {
            return ( (bool)chunk ) && ( chunk_off < chunk_size );
        }

        /*!
         * tries to acquire the element this iterator points to
         * by incrementing the reference count, so it will not be
         * deleted concurrently to the usage of this iterator.
         */
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

        /*!
         * release the storage location
         */
        void release()
        {
            if( has_item )
                item().remove();

            has_item = false;
        }

        /*!
         * advance the position until we find a un-deleted item
         * that is acquired successfully.
         */
        void acquire_next_item()
        {
            while( is_valid_idx() )
            {
                uint16_t step = item().iter_offset;

                if( step == 0 )
                {
                    if( try_acquire() )
                        return;
                    else
                        step = 1;
                }

                if( step <= chunk_off )
                    chunk_off -= step;
                else
                {
                    ++chunk;
                    chunk_off = chunk_size - 1;
                }
            }

            // reached the end here, set chunk-off to invalid idx
            chunk_off = std::numeric_limits< chunk_offset_t >::max();
        }

    public:
        ItemAccess( ItemAccess const & other )
            : ItemAccess( other.chunk_size, other.chunk, other.chunk_off )
        {
        }

        ItemAccess( size_t chunk_size,
                    typename memory::AtomicList< Chunk, Allocator >::MutBackwardIterator chunk,
                    unsigned chunk_off )
            : has_item(false), chunk_size(chunk_size), chunk(chunk), chunk_off(chunk_off)
        {
            acquire_next_item();
        }

        ~ItemAccess()
        {
            release();
        }

        /*! True if the iterator points to a valid storage location,
         * and the item was successfuly locked such that it will not
         * be deleted until this iterator is released.
         */
        bool is_valid() const
        {
            return has_item;
        }

        Item & item() const
        {
            assert( is_valid_idx() );
            return chunk->items()[chunk_off];
        }

        /*! Access item value
         */
        inline
        typename std::conditional<
            is_const,
            T const *,
            T *
        >::type
        operator->() const
        {
            return &item().storage.value;
        }

        inline
        typename std::conditional<
            is_const,
            T const &,
            T &
        >::type
        operator* () const
        {
            return item().storage.value;
        }
    };

    template < bool is_const >
    struct BackwardIterator : ItemAccess< is_const >
    {
        BackwardIterator(
            size_t chunk_size,
            typename memory::AtomicList< Chunk, Allocator >::MutBackwardIterator chunk,
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
    memory::AtomicList< Chunk, Allocator > chunks;

    size_t chunk_size;

public:
    /*
     * @param est_chunk_size gives an estimated number of elements
     *        for each chunk, will be adjusted to make chunks aligned
     */
    ChunkedList( size_t est_chunk_size = 32 )
        : ChunkedList(
            Allocator< uint8_t >(),
            est_chunk_size
        )
    {}

    ChunkedList(
        Allocator< uint8_t > && alloc,
        size_t est_chunk_size = 32
    )
        : chunks(
            std::move(alloc),
            est_chunk_size * sizeof(Item)
        )
    {
        size_t items_capacity = (chunks.get_chunk_capacity() - sizeof(Chunk));
        this->chunk_size = items_capacity / sizeof(Item);
        assert( chunk_size < std::numeric_limits< chunk_offset_t >::max() );
    }

    ChunkedList( ChunkedList && other ) = default;

    ChunkedList( Allocator< uint8_t > && alloc, ChunkedList const & other )
        : ChunkedList( std::move(alloc), other.chunk_size )
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
                    if( chunk_off+1 < chunk_size )
                        chunk->item_count ++;

                    chunk->items()[ chunk_off ] = item;
                    chunk->last_idx ++;
                    return MutBackwardIterator( chunk_size, chunk, chunk_off );
                }
            }

            chunks.allocate_item();
        }
    }

    void remove( MutBackwardIterator const & pos )
    {
        if( pos.is_valid_idx() )
        {
               
            /* first, set iter_offset, so that any iterator
             * will skip this element from now on
             */

            // first elements just goes back one step to reach last element of previous chunk
            if( pos.chunk_off == 0 )
                pos.chunk->items()[ pos.chunk_off ].iter_offset = 1;

            // if we have a predecessor in this chunk, reuse their offset
            else
                pos.chunk->items()[ pos.chunk_off ].iter_offset =
                    pos.chunk->items()[ pos.chunk_off - 1 ].iter_offset + 1;

            /* decrement refcount once so the item will be deconstructed
             * eventually, when all iterators drop their references
             */
            pos.item().remove();

            /* in case all items of this chunk are deleted,
             * delete the chunk too
             */
            if( pos.chunk->item_count.fetch_sub(1) == 1 )
            {
                // spdlog::info("last item!!");
                chunks.erase( pos.chunk );
            }
        }
        else
            throw std::runtime_error("remove invalid position");

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

