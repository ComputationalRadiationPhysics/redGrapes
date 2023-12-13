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
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <redGrapes/util/trace.hpp>
#include <redGrapes/memory/allocator.hpp>
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
    size_t T_chunk_size = 32,
    class Allocator = memory::Allocator
>
struct ChunkedList
{
    using iter_offset_t = uint16_t;
    using refcount_t = int16_t;

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

        /* actual data
         */
        ItemStorage storage;

        /* this variable tells the distance to the next initialized
         * and not already deleted item, where a distance of `0`
         * means that this item exists.
         * In case this item is deleted, `iter_offset` gives an
         * offset by which we can safely jump to find the next
         * existing item.
         *
         * iter_offset = 0 means this item exists
         * iter_offset = 1 means previous item exists
         * ...
         */
        std::atomic< iter_offset_t > iter_offset;

        /* counts the number of iterators pointing
         * at this item currently.
         * It is possible that iterators keep their
         * reference to an item while this item is being
         * deleted. In this case, iter_offset will already
         * be set, so any new iterators will now skip
         * this item but as long as some iterators referencing
         * the already deleted item exist, the item data will
         * not be destructed.
         */
        std::atomic< refcount_t > refcount;

        Item()
            // any item starts uninitialized
            : iter_offset( 1 )
            , refcount( 0 )
            , storage( TrivialInit_t{} )
        {}

        ~Item()
        {
            release<false>();
        }

        /* initialize value of this item.
         * only intended for new elements,
         * re-assigning is not allowed.
         * Per Item, only one thread is allowed to
         * call the assignment operator once.
         */
        T & operator=(T const & value)
        {
            assert( iter_offset != 0 );
            assert( refcount == 0 );

            storage.value = value;

            /* here, item.value is now fully initalized,
             * so allow iterators to access this item now.
             */
            iter_offset = 0;

            return storage.value;
        }

        /* Try to increment `refcount` and check if this
         * item is still alive.
         *
         * @return 0 if acquisition was successful,
         *         otherwise return iterator distance to the next
         *         valid item
         */
        iter_offset_t acquire()
        {
            iter_offset_t off = iter_offset.load();
            refcount_t old_refcount = refcount.load();

            if( iter_offset == 0 && old_refcount >= 0 )
            {
                old_refcount = refcount.fetch_add(1);
                off = iter_offset.load();

                if( old_refcount >= 0 )
                {
                    /* The item data is not already destructed,
                     * but only when `iter_offset` is still set to `0`
                     * as initialized by `operator=`, the item still exists.
                     * In case `off > 0`, some thread already called `remove()`
                     * on this iterator position.
                     */

                    /* keep others from falsely trying to acquire this item
                     * if it is deleted already.
                     */
                    if( off != 0 )
                        --refcount;
                }
                else 
                    /* item data is already destructed.
                     * just decrement refcount to keep others from trying to 
                     * acquire this item.
                     */
                    --refcount;
            }

            return off;
        }

        /* decrement refcount and in case this 
         * was the last reference, deconstruct the element
         * @tparam fail_on_invalid if true, this function will
         *     throw if the item was already deleted
         */
        template < bool fail_on_invalid = true > 
        void release()
        {
            refcount_t old_refcount = refcount.fetch_sub(1);
            if( old_refcount == 0 )
            {
                // item is now deleted, and refcount set to -1
                storage.value.~T();
            }
            else if( old_refcount < 0 )
            {
                if( fail_on_invalid )
                    throw std::runtime_error("ChunkedList: try to remove invalid item!");
            }
        }
    };

    struct Chunk
    {
        /* beginning of the chunk
         */
        std::atomic< Item * > first_item;

        /* points to the latest item which was inserted
         * and is already fully initialized
         */
        std::atomic< Item * > last_item;

        /* points to the next free storage slot,
         * if available. Will be used to add a new element
         */
        std::atomic< Item * > next_item;

        std::atomic< size_t > freed_items{ 0 };

        Chunk( memory::Block blk )
            : first_item( (Item*) blk.ptr )
            , last_item( ((Item*)blk.ptr) - 1 )
            , next_item( (Item*) blk.ptr )
        {
            for(Item * item = this->first_item; item < ( this->first_item + T_chunk_size ); item++ )
                new (item) Item();
        }

        ~Chunk()
        {
            for( Item * item = first_item; item < ( this->first_item + T_chunk_size ); item++ )
                item->~Item();
        }

        Item * items()
        {
            return first_item;
        }
    };

    template < bool is_const >
    struct ItemAccess
    {
    private:
        friend class ChunkedList;       
        typename memory::AtomicList< Chunk, Allocator >::MutBackwardIterator chunk;

        /* this pointer packs the address of the current element
         * and the `has_element` bit in its MSB (most significant bit).
         * Pointers where the MSB is zero indicate an existing storage location
         * but with uninitialized element. Pointers where MSB is set
         * point to an existing element.
         */
        uintptr_t cur_item;

        inline Item * get_item_ptr() const { return (Item *) (cur_item & (~(uintptr_t)0 >> 1)); }
        inline bool has_item() const { return cur_item & ~(~(uintptr_t)0 >> 1); }
        inline void set_item() { cur_item |=  ~( ~(uintptr_t) 0 >> 1 ); }
        inline void unset_item() { cur_item &= ~(uintptr_t)0 >> 1; }

    protected:
        /*!
         * checks whether the iterator points to an existing storage location.
         * This storage location can be used, free or deleted.
         * Only by `rend()`, and if the container is empty also `rbegin()` shall
         * return an iterator with invalid idx.
         */
        bool is_valid_idx() const
        {
            return ((bool)chunk)
                && ( get_item_ptr() >= chunk->first_item )
                && ( get_item_ptr() <= chunk->last_item );
        }

        /*!
         * tries to acquire the element this iterator points to
         * by incrementing the reference count, so it will not be
         * deleted concurrently to the usage of this iterator.
         * @return 0 if acquisition was successful,
         *         otherwise return iterator distance to the next
         *         valid item
         */
        iter_offset_t try_acquire()
        {
            if( is_valid_idx() )
            {
                iter_offset_t off = item().acquire();
                if( off == 0 )
                    set_item();

                return off;
            }
            else
                return 1;
        }

        /*!
         * release the storage location
         */
        void release()
        {
            if( has_item() )
            {
                unset_item();
                item().release();
            }
        }

        /*!
         * advance the position until we find a un-deleted item
         * that is acquired successfully.
         */
        void acquire_next_item()
        {
            while( is_valid_idx() )
            {
                iter_offset_t step = try_acquire();
                if( step == 0 )
                {
                    // item was successfully acquired.
                    assert( has_item() );
                    return;
                }
                else
                {
                    // item is not existent
                    assert( ! has_item() );

                    // jump to next valid item
                    cur_item = (uintptr_t) (get_item_ptr() - step);

                    // goto next chunk if necessary
                    if( ! is_valid_idx() )
                    {
                        ++chunk;
                        if( chunk )
                            cur_item = (uintptr_t) chunk->last_item.load();
                        else
                            cur_item = 0;
                    }
                }
            }

            // reached the end here
            cur_item = 0;
        }

    public:
        ItemAccess( ItemAccess const & other )
            : ItemAccess( other.chunk, other.get_item_ptr() )
        {
        }

        ItemAccess(
            typename memory::AtomicList< Chunk, Allocator >::MutBackwardIterator chunk,
            Item * item_ptr
        )
            : chunk(chunk)
            , cur_item( (uintptr_t)item_ptr )
        {
            acquire_next_item();
        }

        inline ~ItemAccess()
        {
            release();
        }

        /*! True if the iterator points to a valid storage location,
         * and the item was successfuly locked such that it will not
         * be deleted until this iterator is released.
         */
        inline bool is_valid() const
        {
            return has_item();
        }

        inline Item & item() const
        {
            assert( is_valid_idx() );
            return *get_item_ptr();
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
            typename memory::AtomicList< Chunk, Allocator >::MutBackwardIterator chunk,
            Item * start_item
        )
            : ItemAccess< is_const >( chunk, start_item )
        {
        }

        inline bool operator!=(BackwardIterator< is_const > const& other) const
        {
            return this->get_item_ptr() != other.get_item_ptr();
        }

        BackwardIterator< is_const > & operator=( BackwardIterator< is_const > const & other )
        {
            this->release();
            this->cur_item = (uintptr_t) other.get_item_ptr();
            this->chunk = other.chunk;
            this->try_acquire();
            return *this;
        }

        BackwardIterator & operator++()
        {
            this->release();

            if( (uintptr_t)(this->get_item_ptr() - 1u) >= (uintptr_t)this->chunk->first_item.load() )
                this->cur_item = (uintptr_t) (this->get_item_ptr() - 1);
            else
            {
                ++ this->chunk;
                if( this->chunk )
                    this->cur_item = (uintptr_t) this->chunk->last_item.load();
                else
                    this->cur_item = 0;
            }

            this->acquire_next_item();
            return *this;
        }
    };

    using ConstBackwardIterator = BackwardIterator< true >;
    using MutBackwardIterator = BackwardIterator< false >;

private:
    memory::AtomicList< Chunk, Allocator > chunks;

public:
    ChunkedList( Allocator && alloc )
        : chunks( std::move(alloc), T_chunk_size * sizeof(Item) + sizeof(Chunk) )
    {}

    ChunkedList( ChunkedList && other ) = default;
    ChunkedList( Allocator && alloc, ChunkedList const & other )
        : ChunkedList( std::move(alloc) )
    {
        spdlog::error("copy construct ChunkedList!!");
    }

    /* decrement item_count and in case all items of this chunk are deleted,
     * and this chunk is not `head`, delete the chunk too
     */
    void release_chunk( typename memory::AtomicList< Chunk, Allocator >::MutBackwardIterator chunk )
    {
        if( chunk->freed_items.fetch_add(1) == T_chunk_size - 1u )
            chunks.erase( chunk );
    }

    MutBackwardIterator push( T const& item )
    {
        TRACE_EVENT("ChunkedList", "push");

        while( true )
        {
            auto chunk = chunks.rbegin();
            if( chunk != chunks.rend() )
            {
                Item * chunk_begin = chunk->first_item;
                Item * chunk_end = chunk_begin + T_chunk_size;

                /* check if there is a chance to get a slot in this chunk
                 */
                if( (uintptr_t)chunk->next_item.load() <= (uintptr_t)chunk_end )
                {
                    Item * next_item = chunk->next_item.fetch_add(1);

                    if( (uintptr_t)next_item < (uintptr_t)chunk_end )
                    {
                        /* successfully allocated a slot in the current chunk
                         */

                        // initialize item value
                        *next_item = item;

                        /* allow iteration to start at the newly initialized item.
                         * in case it happens that 
                         */
                        chunk->last_item ++;

                        return MutBackwardIterator( chunk, next_item );
                    }
                    else if ( (uintptr_t)next_item == (uintptr_t)chunk_end )
                    {
                        /* here we are the first thread that overflows
                         * the current chunk, so allocate a new chunk here
                         */
                        chunks.allocate_item();
                    }
                    else
                    {
                        /* another one, but not the first thread that overflowed
                         * this chunk. wait for the allocation now.
                         */
                    }
                }
                else
                {
                    /* `chunk` is already full,
                     * don't even attempt to increment `next_item`
                     * just wait for the allocation of the new chunk to happen...
                     */
                }
            }
            else
            {
                chunks.try_allocate_first_item();
            }
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
            if( pos.get_item_ptr() == pos.chunk->first_item )
                pos.item().iter_offset = 1;

            // if we have a predecessor in this chunk, reuse their offset
            else
                pos.item().iter_offset = (pos.get_item_ptr() - 1)->iter_offset + 1;


            /* TODO: scan in other direction for deleted items too,
                and update their `iter_offset` 
             */  

            /* decrement refcount once so the item will be deconstructed
             * eventually, when all iterators drop their references
             */
            pos.item().release();

            release_chunk( pos.chunk );
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
            c,
            ( c != chunks.rend() ) ? c->last_item.load() : nullptr
// TODO: change to this when `last_item` is removed
//            ( c != chunks.rend() ) ? min(c->last_item.load()-1, c->first_item+T_chunk_size) : nullptr
        );
    }

    MutBackwardIterator rend() const
    {
        return MutBackwardIterator(
            chunks.rend(),
            nullptr
        );
    }

    ConstBackwardIterator crbegin() const
    {
        auto c = chunks.rbegin();
        return ConstBackwardIterator(
            c,
            ( c != chunks.rend() ) ? c->last_item.load() : nullptr
        );
    }

    ConstBackwardIterator crend() const
    {
        return ConstBackwardIterator(
            chunks.rend(),
            nullptr
        );
    }
};

} // namespace redGrapes

