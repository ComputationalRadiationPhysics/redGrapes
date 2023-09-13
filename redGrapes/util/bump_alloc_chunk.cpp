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

BumpAllocChunk * alloc_chunk( hwloc_obj_t const & obj, size_t capacity )
{
    size_t alloc_size = capacity + sizeof(BumpAllocChunk);

    BumpAllocChunk * chunk = (BumpAllocChunk*) hwloc_alloc_membind(
        topology, alloc_size, obj->cpuset,
        HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_STRICT
    );

    if( chunk == 0 )
    {
        int error = errno;
        spdlog::error("chunk allocation failed: {}\n", strerror(error));
    }

    new (chunk) BumpAllocChunk( capacity );

    hwloc_set_cpubind(topology, obj->cpuset, HWLOC_CPUBIND_THREAD);

    chunk->reset();

    if( redGrapes::dispatch::thread::current_worker )
        redGrapes::dispatch::thread::current_worker->cpubind();
    else
        cpubind_mainthread();

    return chunk;
}

void free_chunk( hwloc_obj_t const & obj, BumpAllocChunk * chunk )
{
    size_t alloc_size = chunk->capacity + sizeof(BumpAllocChunk);
    hwloc_free( topology, (void*)chunk, alloc_size );
}

BumpAllocChunk::BumpAllocChunk( size_t capacity )
    : capacity( capacity )
    , offset(0)
    , count(0)
{}

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

