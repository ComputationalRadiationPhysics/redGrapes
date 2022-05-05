/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace redGrapes
{
namespace memory
{

struct Chunk
{
    Chunk( size_t capacity);
    Chunk( Chunk & ) = delete;
    ~Chunk();
    
    bool empty() const;
    void reset();

    void * m_alloc( size_t n_bytes );
    void m_free( void * );

private:
    size_t capacity;

    uintptr_t base;
    std::atomic_ptrdiff_t offset;

    std::atomic<unsigned> count;
};

} // namespace memory
} // namespace redGrapes

