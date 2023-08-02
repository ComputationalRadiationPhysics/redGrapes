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
    size_t chunk_size;
    std::shared_ptr< BumpAllocChunk > head;

    ChunkedBumpAlloc( size_t chunk_size )
        : head( std::make_shared<BumpAllocChunk>(chunk_size) )
        , chunk_size( chunk_size )
    {
    }

    inline size_t roundup_to_poweroftwo( size_t s )
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
        T * item = (T*) head->m_alloc( s );

        // chunk is full
        if( !item )
        {
            // create new chunk & try again
            {
                auto new_chunk = std::make_shared<BumpAllocChunk>( chunk_size );
                new_chunk->prev = head;
                std::swap(head, new_chunk);
            }

            item = (T*) head->m_alloc( s );
        }

        return item;
    }

    template <typename T>
    void deallocate( T * ptr )
    {
        std::shared_ptr< BumpAllocChunk > next;
        std::shared_ptr< BumpAllocChunk > cur = head;

        while( cur )
        {
            if( cur->contains((void*)ptr) )
            {
                if( cur->m_free((void*)ptr) == 0 && next )
                {
                    // erase chunk
                    next->prev = cur->prev;
                }
                break;
            }
            else
            {
                next = cur;
                cur = cur->prev;
            }
        }
    }
};

} // namespace memory
} // namespace redGrapes

