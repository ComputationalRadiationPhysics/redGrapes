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

namespace redGrapes
{
namespace memory
{

BumpAllocChunk::BumpAllocChunk( size_t capacity )
    : capacity( capacity )
    , base( (uintptr_t) aligned_alloc(0x80000, capacity) )
{
    reset();
}

BumpAllocChunk::~BumpAllocChunk()
{
    //    free( (void*)base );
}

bool BumpAllocChunk::empty() const
{
    return (count == 0);
}

void BumpAllocChunk::reset()
{
    offset = 0;
    count = 0;
    memset((void*)base, 0, capacity);
}

void * BumpAllocChunk::m_alloc( size_t n_bytes )
{
    std::ptrdiff_t old_offset = offset.fetch_add(n_bytes);
    if( old_offset + n_bytes <= capacity )
    {
        count.fetch_add(1);
        return (void*)(base + old_offset);
    }
    else
        return nullptr;
}

unsigned BumpAllocChunk::m_free( void * )
{
    return count.fetch_sub(1) - 1;
}

bool BumpAllocChunk::contains( void * ptr ) const
{
    return (uintptr_t)ptr >= (uintptr_t)base && (uintptr_t)ptr < (uintptr_t)(base + capacity);
}

} // namespace memory
} // namespace redGrapes

