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

BumpAllocator::BumpAllocator( uintptr_t lower_limit, uintptr_t upper_limit )
    : lower_limit( lower_limit )
    , upper_limit( upper_limit )
    , count(0)
{
    SPDLOG_INFO("bumpallochunk: lower={}, upper={}", lower_limit, upper_limit);
    next_addr = upper_limit;
}

BumpAllocator::~BumpAllocator()
{
#ifndef NDEBUG
    if( !empty() )
        spdlog::warn("BumpAllocChunk: {} allocations remaining not deallocated.", count.load());
#endif
}

bool BumpAllocator::empty() const
{
    return (count == 0);
}

bool BumpAllocator::full() const
{
    return next_addr <= lower_limit;
}

void BumpAllocator::reset()
{
    next_addr = upper_limit;
    count = 0;
}

void * BumpAllocator::allocate( size_t n_bytes )
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

uint16_t BumpAllocator::deallocate( void * )
{
    return count.fetch_sub(1);
}

bool BumpAllocator::contains( void * ptr ) const
{
    uintptr_t p = (uintptr_t)ptr;
    return p >= lower_limit && p < upper_limit;
}

} // namespace memory
} // namespace redGrapes

