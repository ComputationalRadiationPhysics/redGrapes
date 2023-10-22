/* Copyright 2022-2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <boost/core/demangle.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <redGrapes_config.hpp>
#include <spdlog/spdlog.h>
#include <vector>
#include <redGrapes/util/spinlock.hpp>
#include <redGrapes/util/hwloc_alloc.hpp>
#include <redGrapes/util/bump_alloc_chunk.hpp>
#include <redGrapes/util/chunklist.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/dispatch/thread/local.hpp>
#include <redGrapes/dispatch/thread/cpuset.hpp>
#include <redGrapes/util/trace.hpp>
#include <backward.hpp>

namespace redGrapes
{
namespace memory
{

/* use 64KiB as default chunksize
 */
#ifndef REDGRAPES_ALLOC_CHUNKSIZE
#define REDGRAPES_ALLOC_CHUNKSIZE ( 64 * 1024 )
#endif

template < template <typename> class A = HwlocAlloc >
struct ChunkedBumpAlloc
{
    size_t const chunk_size;

    ChunkList< BumpAllocChunk, A > bump_allocators;

    ChunkedBumpAlloc( A<uint8_t> && alloc, size_t chunk_size = REDGRAPES_ALLOC_CHUNKSIZE )
        : chunk_size( chunk_size )
        , bump_allocators( std::move(alloc), chunk_size )
    {
    }

    ChunkedBumpAlloc( ChunkedBumpAlloc && other )
        : chunk_size(other.chunk_size)
        , bump_allocators(other.bump_allocators)
    { 
    }

    ~ChunkedBumpAlloc()
    {
        auto head = bump_allocators.rbegin();
        if( head != bump_allocators.rend() )
            head->count --;
    }

    inline static size_t roundup_to_poweroftwo( size_t s )
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

    template <typename T>
    T * allocate( std::size_t n = 1 )
    {
        TRACE_EVENT("Allocator", "ChunkedBumpAlloc::allocate()");
        size_t alloc_size = roundup_to_poweroftwo( n * sizeof(T) );
 
        size_t const chunk_capacity = bump_allocators.get_chunk_capacity();

        if( alloc_size <= chunk_capacity )
        {
            T * item = (T*) nullptr;

            while( !item )
            {
                // try to alloc in current chunk
                auto chunk = bump_allocators.rbegin();

                if( chunk != bump_allocators.rend() )
                {
                    item = (T*) chunk->m_alloc( alloc_size );

                    // chunk is full, create a new one
                    if( !item )
                    {
                        bump_allocators.add_chunk();

                        /* now, that the new chunk is added,
                         * decrease count of old block so it can be deleted
                         * by any following `deallocate`
                         */
                        if( chunk->count.fetch_sub(1) == 1 )
                            bump_allocators.erase( chunk );
                    }
                }
                // no chunk exists, create a new one
                else
                    bump_allocators.add_chunk();
            }

            SPDLOG_TRACE("ChunkedBumpAlloc: alloc {},{}", (uintptr_t)item, alloc_size);
            return item;
        }
        else
        {
            spdlog::error("ChunkedBumpAlloc: requested allocation of {} bytes exceeds chunk capacity of {} bytes", alloc_size, chunk_capacity);
            return nullptr;
        }
    }

    template < typename T >
    void deallocate( T * ptr )
    {
        TRACE_EVENT("Allocator", "ChunkedBumpAlloc::deallocate()");
        SPDLOG_TRACE("ChunkedBumpAlloc[{}]: free {} ", (void*)this, (uintptr_t)ptr);

        /* find the chunk that contains `ptr` and deallocate there.
         * Additionally, delete the chunk if possible.
         */

        auto prev = bump_allocators.rbegin();
        for( auto it = bump_allocators.rbegin(); it != bump_allocators.rend(); ++it )
        {
            if( it->contains((void*) ptr) )
            {
                /* if no allocations remain in this chunk
                 * and this chunk is not `head`,
                 * remove this chunk
                 */
                if( it->m_free((void*)ptr) == 1 )
                {
                    SPDLOG_TRACE("ChunkedBumpAlloc: erase chunk {}", it->lower_limit);
                    bump_allocators.erase( it );
                    prev.optimize();
                }

                return;
            }
            prev = it;
        }

        spdlog::error("try to deallocate invalid pointer ({}). current_arena={}, this={}", (void*)ptr, current_arena, (void*)this);

        backward::StackTrace st;
        st.load_here(32);
        backward::Printer p;
        p.print(st);
    }
};

} // namespace memory
} // namespace redGrapes

