/* Copyright 2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <redGrapes/memory/allocator.hpp>
#include <redGrapes/memory/block.hpp>
#include <redGrapes/memory/refcounted.hpp>
#include <redGrapes/util/trace.hpp>

#include <fmt/format.h>
#include <hwloc.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace redGrapes
{
    namespace memory
    {


        /* maintains a lockfree singly-linked list
         * with the following allowed operations:
         *   - append new chunks at head
         *   - erase any chunk which is not current head
         *   - reversed iteration (starting at head)
         *
         * each chunk is managed through a `std::shared_ptr` which points to a
         * contiguous block containing list-metadata, the chunk-control-object
         * (`ChunkData`) and freely usable data.
         *
         * @tparam Item element type
         * @tparam Allocator must satisfy `Allocator` concept
         */
        template<typename Item, typename Allocator>
        struct AtomicList
        {
            struct ItemControlBlock;

            struct ItemControlBlockDeleter
            {
                void operator()(ItemControlBlock*);
            };

            struct ItemControlBlock : Refcounted<ItemControlBlock, ItemControlBlockDeleter>
            {
                using Guard = typename Refcounted<ItemControlBlock, ItemControlBlockDeleter>::Guard;

                Guard prev;
                uintptr_t item_data_ptr;
                Allocator alloc;

                template<typename... Args>
                ItemControlBlock(Allocator alloc, memory::Block blk)
                    : prev(nullptr)
                    , item_data_ptr(blk.ptr)
                    , alloc(alloc)
                {
                    /* put Item at front and initialize it
                     * with the remaining memory region
                     */
                    blk.ptr += sizeof(Item);
                    blk.len -= sizeof(Item);
                    new(get()) Item(blk);
                }

                ~ItemControlBlock()
                {
                    get()->~Item();
                }

                /* flag this chunk as deleted and call ChunkData destructor
                 */
                inline void erase()
                {
                    // set MSB (most significant bit) of item_data ptr
                    item_data_ptr |= ~(~(uintptr_t) 0 >> 1);
                }

                inline bool is_deleted() const
                {
                    return item_data_ptr & ~(~(uintptr_t) 0 >> 1);
                }

                inline Item* get() const
                {
                    return (Item*) (item_data_ptr & (~(uintptr_t) 0 >> 1));
                }

                /* adjusts `prev` so that it points to a non-deleted chunk again
                 * this should free the shared_ptr of the original prev
                 * in case no iterators point to it
                 */
                void skip_deleted_prev()
                {
                    Guard p = prev;
                    while(p && p->is_deleted())
                        p = p->prev;

                    prev = p;
                }
            };

            typename ItemControlBlock::Guard head;
            size_t const chunk_capacity;
            Allocator alloc;

        public:
            AtomicList(Allocator&& alloc, size_t chunk_capacity)
                : alloc(alloc)
                , head(nullptr)
                , chunk_capacity(chunk_capacity)
            {
            }

            static constexpr size_t get_controlblock_size()
            {
                return sizeof(ItemControlBlock) + sizeof(Item);
            }

            constexpr size_t get_chunk_capacity()
            {
                return chunk_capacity;
            }

            constexpr size_t get_chunk_allocsize()
            {
                return chunk_capacity + get_controlblock_size();
            }

            /* allocate a new item and add it to the list
             *
             * @{
             */
            auto allocate_item()
            {
                TRACE_EVENT("Allocator", "AtomicList::allocate_item()");

                memory::Block blk = this->alloc.allocate(get_chunk_allocsize());
                ItemControlBlock* item_ctl = (ItemControlBlock*) blk.ptr;

                blk.ptr += sizeof(ItemControlBlock);
                blk.len -= sizeof(ItemControlBlock);

                new(item_ctl) ItemControlBlock(alloc, blk);
                return append_item(std::move(typename ItemControlBlock::Guard(item_ctl)));
            }

            /** allocate the first item if the list is empty
             *
             * If more than one thread tries to add the first item only one thread will successfully add an item.
             */
            bool try_allocate_first_item()
            {
                TRACE_EVENT("Allocator", "AtomicList::try_allocate_first_item()");

                memory::Block blk = this->alloc.allocate(get_chunk_allocsize());
                ItemControlBlock* item_ctl = (ItemControlBlock*) blk.ptr;

                blk.ptr += sizeof(ItemControlBlock);
                blk.len -= sizeof(ItemControlBlock);

                new(item_ctl) ItemControlBlock(alloc, blk);
                return try_append_first_item(std::move(typename ItemControlBlock::Guard(item_ctl)));
            }

            /** @} */

            template<bool is_const = false>
            struct BackwardIterator
            {
                typename ItemControlBlock::Guard c;

                void erase()
                {
                    c->erase();
                }

                bool operator!=(BackwardIterator const& other) const
                {
                    return c != other.c;
                }

                operator bool() const
                {
                    return (bool) c;
                }

                typename std::conditional<is_const, Item const*, Item*>::type operator->() const
                {
                    return c->get();
                }

                typename std::conditional<is_const, Item const&, Item&>::type operator*() const
                {
                    return *c->get();
                }

                void optimize()
                {
                    if(c)
                        c->skip_deleted_prev();
                }

                BackwardIterator& operator++()
                {
                    if(c)
                    {
                        c->skip_deleted_prev();
                        c = c->prev;
                    }

                    return *this;
                }
            };

            using ConstBackwardIterator = BackwardIterator<true>;
            using MutBackwardIterator = BackwardIterator<false>;

            /* get iterator starting at current head, iterating backwards from
             * most recently added to least recently added
             */
            MutBackwardIterator rbegin() const
            {
                return MutBackwardIterator{typename ItemControlBlock::Guard(head)};
            }

            MutBackwardIterator rend() const
            {
                return MutBackwardIterator{typename ItemControlBlock::Guard()};
            }

            ConstBackwardIterator crbegin() const
            {
                return ConstBackwardIterator{typename ItemControlBlock::Guard(head)};
            }

            ConstBackwardIterator crend() const
            {
                return ConstBackwardIterator{typename ItemControlBlock::Guard()};
            }

            /* Flags chunk at `pos` as erased. Actual removal is delayed until
             * iterator stumbles over it.
             *
             * Since we only append to the end and `chunk` is not `head`,
             * there wont occur any inserts after this chunk.
             */
            void erase(MutBackwardIterator pos)
            {
                pos.erase();
            }

            /* atomically appends a floating chunk to this list
             * and returns the previous head to which the new_head
             * is now linked.
             */
            auto append_item(typename ItemControlBlock::Guard new_head)
            {
                TRACE_EVENT("Allocator", "AtomicList::append_item()");
                typename ItemControlBlock::Guard old_head;

                bool append_successful = false;
                while(!append_successful)
                {
                    typename ItemControlBlock::Guard old_head(head);
                    new_head->prev = old_head;
                    append_successful = head.compare_exchange_strong(old_head.get(), new_head);
                }

                return MutBackwardIterator{old_head};
            }

            // append the first head item if not already exists
            bool try_append_first_item(typename ItemControlBlock::Guard new_head)
            {
                TRACE_EVENT("Allocator", "AtomicList::append_first_item()");

                return head.compare_exchange_strong(nullptr, new_head);
            }
        };

        template<typename Item, typename Allocator>
        void AtomicList<Item, Allocator>::ItemControlBlockDeleter::operator()(
            AtomicList<Item, Allocator>::ItemControlBlock* e)
        {
            Allocator alloc = e->alloc;
            e->~ItemControlBlock();
            memory::Block blk{(uintptr_t) e, 0};
            alloc.deallocate(blk);
        }

    } // namespace memory

} // namespace redGrapes
