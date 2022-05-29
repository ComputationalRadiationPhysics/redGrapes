/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cstdlib>
#include <atomic>
#include <spdlog/spdlog.h>

#include <redGrapes/task/allocator.hpp>

namespace redGrapes
{
namespace memory
{

Chunk::Chunk( size_t capacity )
    : base( (uintptr_t) malloc(capacity) )
    , capacity( capacity )
{
    reset();
}

Chunk::~Chunk()
{
    free( (void*)base );
}

bool Chunk::empty() const
{
    return (count == 0);
}

void Chunk::reset()
{
    offset = 0;
    count = 0;
    memset((void*)base, 0, capacity);
}

void * Chunk::m_alloc( size_t n_bytes )
{
    std::ptrdiff_t old_offset = offset.fetch_add(n_bytes);
    if( old_offset + n_bytes <= capacity )
    {
        count++;
        return (void*)(base + old_offset);
    }
    else
        return nullptr;
}

void Chunk::m_free( void * )
{
    count--;
}

bool Chunk::contains( void * ptr )
{
    return (uintptr_t)ptr >= (uintptr_t)base && (uintptr_t)ptr < (uintptr_t)(base + capacity);
}

} // namespace memory
} // namespace redGrapes

