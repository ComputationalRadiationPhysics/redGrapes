/* Copyright 2023-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include "redGrapes/memory/block.hpp"

#include <spdlog/spdlog.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace redGrapes
{
    namespace memory
    {

        /* The `BumpAllocator` manages a chunk of memory,
         * given by `lower_limit` and `upper_limit` by
         * decrementing the `next_addr` by the requested size,
         * and counting the number of active allocations.
         * The data will start immediately after this management object
         */
        struct BumpAllocator
        {
            BumpAllocator(Block blk) : BumpAllocator((uintptr_t) blk.ptr, (uintptr_t) blk.ptr + blk.len)
            {
            }

            BumpAllocator(uintptr_t lower_limit, uintptr_t upper_limit)
                : count(0)
                , lower_limit(lower_limit)
                , upper_limit(upper_limit)
            {
                spdlog::debug("bumpallochunk: lower={}, upper={}", lower_limit, upper_limit);
                next_addr = upper_limit;
            }

            BumpAllocator(BumpAllocator const&) = delete;
            BumpAllocator(BumpAllocator&) = delete;

            ~BumpAllocator()
            {
#ifndef NDEBUG
                if(!empty())
                    spdlog::warn("BumpAllocChunk: {} allocations remaining not deallocated.", count.load());
#endif
            }

            void reset()
            {
                next_addr = upper_limit;
                count = 0;
            }

            bool empty() const
            {
                return (count == 0);
            }

            /* check whether this allocator is exhausted already.
             * @return true if no free space remains
             */
            bool full() const
            {
                return next_addr <= lower_limit;
            }

            /*! checks whether this block is managed by this allocator
             */
            bool owns(Block const& blk) const
            {
                return blk.ptr >= lower_limit && blk.ptr < upper_limit;
            }

            /*! @param n_bytes size of requested memory block
             * @return Block with len = n_bytes and some non-nullpointer
             *         if successful, return Block::null() on exhaustion.
             */
            Block allocate(size_t n_bytes)
            {
                uintptr_t addr = next_addr.fetch_sub(n_bytes) - n_bytes;
                if(addr >= lower_limit)
                {
                    count++;
                    return Block{addr, n_bytes};
                }
                else
                    return Block::null();
            }

            /*! @return how many active allocations remain,
             * if it returns 0, this allocator needs to be reset.
             */
            uint16_t deallocate(Block blk)
            {
                assert(owns(blk));
                return count.fetch_sub(1);
            }


        private:
            //! pointer to the upper-limit of the next allocation
            std::atomic<uintptr_t> next_addr;

            //! number of active allocations
            std::atomic<uint16_t> count;
            uintptr_t const lower_limit;
            uintptr_t const upper_limit;
        };

    } // namespace memory
} // namespace redGrapes
