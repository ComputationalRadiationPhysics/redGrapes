/* Copyright 2022 Michael Sippel
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
#include <redGrapes/util/bump_alloc_chunk.hpp>
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
    hwloc_obj_t const obj;

    std::atomic< BumpAllocChunk * > head;

    ChunkedBumpAlloc( hwloc_obj_t obj, size_t chunk_size )
        : obj( obj )
        , chunk_size( chunk_size )
        , head( alloc_chunk( obj, chunk_size ) )
    {
    }

    ChunkedBumpAlloc( ChunkedBumpAlloc && other )
        : obj( other.obj )
        , chunk_size( other.chunk_size )
        , head( other.head.load() )
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

        // try to alloc in current chunk
        T * item = (T*) head.load()->m_alloc( s );

        // chunk is full
        if( !item )
        {
            // create new chunk & try again
            {
                BumpAllocChunk * old_chunk = head.load();
                BumpAllocChunk * new_chunk = alloc_chunk( obj, chunk_size );
                new_chunk->prev = old_chunk;

                head.compare_exchange_strong( old_chunk, new_chunk );
            }

            item = (T*) head.load()->m_alloc( s );
        }

        return item;
    }

    template <typename T>
    void deallocate( T * ptr )
    {
        BumpAllocChunk * next = nullptr;
        BumpAllocChunk * cur = head.load();

        while( cur )
        {
            if( cur->contains((void*)ptr) )
            {
                // if no allocations remain in this chunk
                // and this chunk is not the latest one,
                // remove this chunk
                if( cur->m_free((void*)ptr) == 0 && next )
                {
                    // erase from linked list
                    next->prev.compare_exchange_strong( cur, cur->prev );

                    // free memory
                    free_chunk( obj, cur );
                }
                break;
            }
            else
            {
                next = cur;
                cur = cur->prev.load();
            }
        }
    }
};

} // namespace memory
} // namespace redGrapes

