/* Copyright 2023 The RedGrapes Community
 *
 * Authors: Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <redGrapes/memory/block.hpp>

#include <atomic>
#include <cstdint>

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
            BumpAllocator(Block blk);
            BumpAllocator(uintptr_t lower_limit, uintptr_t upper_limit);
            BumpAllocator(BumpAllocator const&) = delete;
            BumpAllocator(BumpAllocator&) = delete;
            ~BumpAllocator();

            void reset();

            bool empty() const;

            /* check whether this allocator is exhausted already.
             * @return true if no free space remains
             */
            bool full() const;

            /*! checks whether this block is managed by this allocator
             */
            bool owns(Block const&) const;

            /*! @param n_bytes size of requested memory block
             * @return Block with len = n_bytes and some non-nullpointer
             *         if successful, return Block::null() on exhaustion.
             */
            Block allocate(size_t n_bytes);

            /*! @return how many active allocations remain,
             * if it returns 0, this allocator needs to be reset.
             */
            uint16_t deallocate(Block blk);

        private:
            //! number of active allocations
            std::atomic<uint16_t> count;

            //! pointer to the upper-limit of the next allocation
            std::atomic<uintptr_t> next_addr;


            uintptr_t const lower_limit;
            uintptr_t const upper_limit;
        };

    } // namespace memory
} // namespace redGrapes
