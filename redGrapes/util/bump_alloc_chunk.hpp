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

namespace redGrapes
{
namespace memory
{

/* A chunk of memory, inside of which bump allocation is used
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

private:
    // next address that will be allocated
    std::atomic_ptrdiff_t offset;

    // max. size of chunk in bytes
    size_t capacity;

    // start address of chunk
    uintptr_t base;

    // number of allocations
    std::atomic<unsigned> count;

public:
    // reference to previously filled chunks,
    std::shared_ptr< BumpAllocChunk > prev;
};



} // namespace memory
} // namespace redGrapes

