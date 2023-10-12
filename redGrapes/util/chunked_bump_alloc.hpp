/* Copyright 2022-2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <redGrapes/util/spinlock.hpp>
#include <redGrapes/util/hwloc_alloc.hpp>
#include <redGrapes/util/bump_alloc_chunk.hpp>
#include <redGrapes/util/chunklist.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/dispatch/thread/local.hpp>
#include <redGrapes/dispatch/thread/cpuset.hpp>

namespace redGrapes
{
namespace memory
{

struct ChunkedBumpAlloc
{
    size_t const chunk_size;

    ChunkList< BumpAllocChunk, HwlocAlloc > bump_allocators;

    ChunkedBumpAlloc( hwloc_obj_t obj, size_t chunk_size )
        : chunk_size(chunk_size)
        , bump_allocators( HwlocAlloc<uint8_t>(obj), sizeof(BumpAllocChunk) + chunk_size )
    {}

    ChunkedBumpAlloc( ChunkedBumpAlloc && other )
        : chunk_size(other.chunk_size)
        , bump_allocators( other.bump_allocators )
    {        
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
        size_t s = n * sizeof(T);
        s = roundup_to_poweroftwo(s);

        if( s <= chunk_size )
        {
            // try to alloc in current chunk
            T * item = (T*) nullptr;

            // chunk is full, create a new one
            while( !item )
            {
                auto chunk = bump_allocators.rbegin();

                if( chunk != bump_allocators.rend() )
                {
                    item = (T*) chunk->m_alloc( s );
                    if( !item )
                        bump_allocators.add_chunk( chunk_size );
                }
                else
                    bump_allocators.add_chunk( chunk_size );
            }

            return item;
        }
        else
        {
            spdlog::error("ChunkedBumpAlloc: requested allocation of {} bytes exceeds chunksize of {} bytes", s, chunk_size);
            throw std::bad_alloc();
        }
    }

    template <typename T>
    void deallocate( T * ptr )
    {
        for( auto it = bump_allocators.rbegin(); it != bump_allocators.rend(); ++it )
        {
            if( it->contains((void*) ptr) )
            {
                // if no allocations remain in this chunk
                // and this chunk is not the latest one,
                // remove this chunk
                if( it->m_free((void*)ptr) == 0 )
                    bump_allocators.erase( it );

                return;
            }
        }
    }
};

} // namespace memory
} // namespace redGrapes

