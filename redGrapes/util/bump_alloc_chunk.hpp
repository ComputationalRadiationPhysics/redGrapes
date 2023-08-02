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

namespace redGrapes
{

extern hwloc_topology_t topology;

namespace memory
{

/* A chunk of memory, inside of which bump allocation is used.
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

    // reference to previously filled chunks,
    std::atomic< BumpAllocChunk * > prev;

    // max. size of chunk in bytes
    size_t const capacity;

    // next address that will be allocated
    std::atomic_ptrdiff_t offset;

    // number of allocations
    std::atomic<unsigned> count;
};

BumpAllocChunk * alloc_chunk( hwloc_obj_t const & obj, size_t capacity );
void free_chunk( hwloc_obj_t const & obj, BumpAllocChunk * chunk );

} // namespace memory
} // namespace redGrapes

