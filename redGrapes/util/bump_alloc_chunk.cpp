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

BumpAllocChunk * alloc_chunk( hwloc_obj_t const & obj, size_t capacity )
{
    size_t alloc_size = capacity + sizeof(BumpAllocChunk);

    BumpAllocChunk * chunk = (BumpAllocChunk*) hwloc_alloc_membind(
        topology, alloc_size, obj->cpuset,
        HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD
    );

    new (chunk) BumpAllocChunk( capacity );

    return chunk;
}

void free_chunk( hwloc_obj_t const & obj, BumpAllocChunk * chunk )
{
    size_t alloc_size = chunk->capacity + sizeof(BumpAllocChunk);
    hwloc_free( topology, (void*)chunk, alloc_size );
}


BumpAllocChunk::BumpAllocChunk( size_t capacity )
    : capacity( capacity )
{
    reset();
}

BumpAllocChunk::~BumpAllocChunk()
{
}

uintptr_t BumpAllocChunk::get_baseptr() const
{
    return (uintptr_t)this + sizeof(BumpAllocChunk);
}

bool BumpAllocChunk::empty() const
{
    return (count == 0);
}

void BumpAllocChunk::reset()
{
    offset = 0;
    count = 0;
    memset((void*)get_baseptr(), 0, capacity);
}

void * BumpAllocChunk::m_alloc( size_t n_bytes )
{
    std::ptrdiff_t old_offset = offset.fetch_add(n_bytes);
    if( old_offset + n_bytes <= capacity )
    {
        count.fetch_add(1);
        return (void*)(get_baseptr() + old_offset);
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
    return (uintptr_t)ptr >= (uintptr_t)get_baseptr() && (uintptr_t)ptr < (uintptr_t)(get_baseptr() + capacity);
}

} // namespace memory
} // namespace redGrapes

