/* Copyright 2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>

#include <hwloc.h>
#include <spdlog/spdlog.h>

namespace redGrapes
{
namespace memory
{

/* A chunk of memory, inside of which bump allocation is performed.
 * The data will start immediately after this management object
 */
struct BumpAllocChunk
{
    BumpAllocChunk( size_t capacity );
    BumpAllocChunk( BumpAllocChunk const & ) = delete;
    BumpAllocChunk( BumpAllocChunk & ) = delete;
    ~BumpAllocChunk();

    bool empty() const;
    void reset();

    void * m_alloc( size_t n_bytes );
    unsigned m_free( void * );

    bool contains( void * ) const;

    uintptr_t get_baseptr() const;

    // max. size of chunk in bytes
    size_t const capacity;

    // next address that will be allocated
    std::atomic_ptrdiff_t offset;

    // number of allocations
    std::atomic<unsigned> count;    
};

template < template <typename> typename Allocator >
BumpAllocChunk * alloc_chunk( Allocator< uint8_t > & alloc, size_t capacity )
{
    size_t alloc_size = capacity + sizeof(BumpAllocChunk);
    BumpAllocChunk * chunk = (BumpAllocChunk*) alloc.allocate( alloc_size );

    if( chunk == 0 )
    {
        int error = errno;
        spdlog::error("chunk allocation failed: {}\n", strerror(error));
    }

    new (chunk) BumpAllocChunk( capacity );

    return chunk;
}

template < template <typename> typename Allocator >
void free_chunk( Allocator< uint8_t > & alloc, BumpAllocChunk * chunk )
{
    alloc.deallocate( chunk );
}

} // namespace memory
} // namespace redGrapes

