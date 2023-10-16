/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cstdlib>
#include <atomic>
#include <redGrapes/util/bump_alloc_chunk.hpp>
#include <cstring>
#include <redGrapes/dispatch/thread/worker.hpp>
#include <redGrapes/redGrapes.hpp>

namespace redGrapes
{
namespace memory
{

BumpAllocChunk::BumpAllocChunk( size_t capacity )
    : lower_limit( (uintptr_t)this + sizeof(BumpAllocChunk) )
    , upper_limit( lower_limit + capacity )
    , count(1)
{
    next_addr = upper_limit;
}

BumpAllocChunk::~BumpAllocChunk()
{
    if( !empty() )
        spdlog::warn("BumpAllocChunk: {} allocations remaining not deallocated.", count.load());
}

bool BumpAllocChunk::empty() const
{
    return (count == 0);
}

void BumpAllocChunk::reset()
{
    next_addr = upper_limit;
    count = 0;
    memset((void*) lower_limit, 0, upper_limit-lower_limit);
}

void * BumpAllocChunk::m_alloc( size_t n_bytes )
{
    uintptr_t addr = next_addr.fetch_sub( n_bytes ) - n_bytes;
    if( addr >= lower_limit )
    {
        count ++;
        return (void*)addr;
    }
    else
        return nullptr;
}

size_t BumpAllocChunk::m_free( void * )
{
    return count.fetch_sub(1) - 1;
}

bool BumpAllocChunk::contains( void * ptr ) const
{
    uintptr_t p = (uintptr_t)ptr;
    return p >= lower_limit && p < upper_limit;
}

} // namespace memory
} // namespace redGrapes

