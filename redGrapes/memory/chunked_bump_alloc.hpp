/* Copyright 2022-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/memory/bump_allocator.hpp"
#include "redGrapes/memory/hwloc_alloc.hpp"
#include "redGrapes/util/atomic_list.hpp"
#include "redGrapes/util/trace.hpp"

#include <boost/core/demangle.hpp>
#include <spdlog/spdlog.h>

#include <cstddef>

#if REDGRAPES_ENABLE_BACKWARDCPP
#    include <backward.hpp>
#endif

namespace redGrapes
{
    namespace memory
    {

/* use 64KiB as default chunksize
 */
#ifndef REDGRAPES_ALLOC_CHUNKSIZE
#    define REDGRAPES_ALLOC_CHUNKSIZE (64 * 1024)
#endif

        struct HwlocAlloc;

        template<typename Alloc = HwlocAlloc>
        struct ChunkedBumpAlloc
        {
            size_t const chunk_size;

            AtomicList<BumpAllocator, Alloc> bump_allocators;

            ChunkedBumpAlloc(Alloc&& alloc, size_t chunk_size = REDGRAPES_ALLOC_CHUNKSIZE)
                : chunk_size(chunk_size)
                , bump_allocators(std::move(alloc), chunk_size)
            {
            }

            static inline size_t roundup_to_poweroftwo(size_t s)
            {
                s--;
                s |= s >> 0x1;
                s |= s >> 0x2;
                s |= s >> 0x4;
                s |= s >> 0x8;
                s |= s >> 0x10;
                s |= s >> 0x20;
                s++;
                return s;
            }

            Block allocate(std::size_t n = 1) noexcept
            {
                TRACE_EVENT("Allocator", "ChunkedBumpAlloc::allocate()");
                size_t alloc_size = roundup_to_poweroftwo(n);

                size_t const chunk_capacity = bump_allocators.get_chunk_capacity();

                if(alloc_size <= chunk_capacity)
                {
                    Block blk = Block::null();

                    while(!blk)
                    {
                        // try to alloc in current chunk
                        auto chunk = bump_allocators.rbegin();

                        if(chunk != bump_allocators.rend())
                        {
                            blk = chunk->allocate(alloc_size);

                            // chunk is full, create a new one
                            if(!blk)
                                bump_allocators.allocate_item();
                        }
                        // no chunk exists, create a new one
                        else
                            bump_allocators.allocate_item();
                    }

                    SPDLOG_TRACE("ChunkedBumpAlloc: alloc {},{}", blk.ptr, blk.len);
                    return blk;
                }
                else
                {
                    SPDLOG_ERROR(
                        "ChunkedBumpAlloc: requested allocation of {} bytes exceeds chunk capacity of {} bytes",
                        alloc_size,
                        chunk_capacity);
                    return Block::null();
                }
            }

            void deallocate(Block blk)
            {
                TRACE_EVENT("Allocator", "ChunkedBumpAlloc::deallocate()");
                SPDLOG_TRACE("ChunkedBumpAlloc[{}]: free {} {} ", (void*) this, (uintptr_t) blk.ptr, blk.len);

                /* find the chunk that contains `ptr` and deallocate there.
                 * Additionally, delete the chunk if possible.
                 */

                auto prev = bump_allocators.rbegin();
                for(auto it = bump_allocators.rbegin(); it != bump_allocators.rend(); ++it)
                {
                    if(it->owns(blk))
                    {
                        /* if no allocations remain in this chunk
                         * and this chunk is not `head`,
                         * remove this chunk
                         */
                        if(it->deallocate(blk) == 1)
                        {
                            SPDLOG_TRACE("ChunkedBumpAlloc: erase chunk");
                            if(it->full())
                            {
                                bump_allocators.erase(it);
                                prev.optimize();
                            }
                        }

                        return;
                    }
                    prev = it;
                }

#if REDGRAPES_ENABLE_BACKWARDCPP
                SPDLOG_ERROR("try to deallocate invalid pointer ({}). this={}", (void*) blk.ptr, (void*) this);

                backward::StackTrace st;
                st.load_here(32);
                backward::Printer p;
                p.print(st);
#endif
            }
        };

    } // namespace memory
} // namespace redGrapes
