/* Copyright 2023-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/memory/block.hpp"
#include "redGrapes/util/demangled_name.hpp"
#include "redGrapes/util/trace.hpp"

#include <fmt/format.h>
#include <hwloc.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

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
            // private:
            struct ItemControlBlock
            {
                bool volatile deleted;
                std::shared_ptr<ItemControlBlock> prev;
                uintptr_t item_data_ptr;

                ItemControlBlock(memory::Block blk) : deleted(false), item_data_ptr(blk.ptr)
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
                void erase()
                {
                    deleted = true;
                }

                /* adjusts `prev` so that it points to a non-deleted chunk again
                 * this should free the shared_ptr of the original prev
                 * in case no iterators point to it
                 */
                void skip_deleted_prev()
                {
                    std::shared_ptr<ItemControlBlock> p = std::atomic_load(&prev);
                    while(p && p->deleted)
                        p = std::atomic_load(&p->prev);

                    std::atomic_store(&prev, p);
                }

                Item* get() const
                {
                    return (Item*) item_data_ptr;
                }
            };

            Allocator alloc;
            std::shared_ptr<ItemControlBlock> head;
            size_t const chunk_capacity;

            /* keeps a single, predefined pointer
             * and frees it on deallocate.
             * used to spoof the allocated size to be bigger than requested.
             */
            template<typename T>
            struct StaticAlloc
            {
                typedef T value_type;

                Allocator alloc;
                T* ptr;

                StaticAlloc(Allocator alloc, size_t n_bytes) : alloc(alloc), ptr((T*) alloc.allocate(n_bytes))
                {
                    SPDLOG_TRACE(" object: {} , bytes : {}", util::type_name<Item>(), n_bytes);
                }

                template<typename U>
                constexpr StaticAlloc(StaticAlloc<U> const& other) noexcept : alloc(other.alloc)
                                                                            , ptr((T*) other.ptr)
                {
                }

                T* allocate(size_t n) noexcept
                {
                    return ptr;
                }

                void deallocate(T* p, std::size_t n) noexcept
                {
                    alloc.deallocate(Block{.ptr = (uintptr_t) p, .len = sizeof(T) * n});
                }
            };

        public:
            AtomicList(Allocator&& alloc, size_t chunk_capacity)
                : alloc(alloc)
                , head(nullptr)
                , chunk_capacity(chunk_capacity)
            {
            }

            static constexpr size_t get_controlblock_size()
            {
                /* TODO: use sizeof( ...shared_ptr_inplace_something... )
                 */
                size_t const shared_ptr_size = 512;
                return sizeof(ItemControlBlock) + shared_ptr_size;
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

                /* NOTE: we are relying on std::allocate_shared
                 * to do one *single* allocation which contains:
                 * - shared_ptr control block
                 * - chunk control block
                 * - chunk data
                 * whereby chunk data is not included by sizeof(ItemControlBlock),
                 * but reserved by StaticAlloc.
                 * This works because shared_ptr control block lies at lower address.
                 */
                StaticAlloc<void> chunk_alloc(this->alloc, get_chunk_allocsize());

                // this block will contain the Item-data of ItemControlBlock
                memory::Block blk{
                    .ptr = (uintptr_t) chunk_alloc.ptr + get_controlblock_size(),
                    .len = chunk_capacity - get_controlblock_size()};

                return append_item(std::allocate_shared<ItemControlBlock>(chunk_alloc, blk));
            }

            /** allocate the first item if the list is empty
             *
             * If more than one thread tries to add the first item only one thread will successfully add an item.
             */
            bool try_allocate_first_item()
            {
                TRACE_EVENT("Allocator", "AtomicList::allocate_first_item()");
                StaticAlloc<void> chunk_alloc(this->alloc, get_chunk_allocsize());

                // this block will contain the Item-data of ItemControlBlock
                memory::Block blk{
                    .ptr = (uintptr_t) chunk_alloc.ptr + get_controlblock_size(),
                    .len = chunk_capacity - get_controlblock_size()};

                auto sharedChunk = std::allocate_shared<ItemControlBlock>(chunk_alloc, blk);
                return try_append_first_item(std::move(sharedChunk));
            }

            /** @} */

            template<bool is_const = false>
            struct BackwardIterator
            {
                std::shared_ptr<ItemControlBlock> c;

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
                return MutBackwardIterator{std::atomic_load(&head)};
            }

            MutBackwardIterator rend() const
            {
                return MutBackwardIterator{std::shared_ptr<ItemControlBlock>()};
            }

            ConstBackwardIterator crbegin() const
            {
                return ConstBackwardIterator{std::atomic_load(&head)};
            }

            ConstBackwardIterator crend() const
            {
                return ConstBackwardIterator{std::shared_ptr<ItemControlBlock>()};
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
            auto append_item(std::shared_ptr<ItemControlBlock> const& new_head)
            {
                TRACE_EVENT("Allocator", "AtomicList::append_item()");
                std::shared_ptr<ItemControlBlock> old_head;

                bool append_successful = false;
                while(!append_successful)
                {
                    old_head = std::atomic_load(&head);
                    std::atomic_store(&new_head->prev, old_head);
                    append_successful = std::atomic_compare_exchange_strong(&head, &old_head, new_head);
                }

                return MutBackwardIterator{old_head};
            }

            // append the first head item if not already exists
            bool try_append_first_item(std::shared_ptr<ItemControlBlock> const& new_head)
            {
                TRACE_EVENT("Allocator", "AtomicList::append_first_item()");

                std::shared_ptr<ItemControlBlock> expected(nullptr);
                std::shared_ptr<ItemControlBlock> const& desired = new_head;
                return std::atomic_compare_exchange_strong(&head, &expected, desired);
            }
        };

    } // namespace memory

} // namespace redGrapes
