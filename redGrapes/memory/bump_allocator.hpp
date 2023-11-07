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
struct BumpAllocator
{
    BumpAllocator( uintptr_t lower_limit, uintptr_t upper_limit );
    BumpAllocator( BumpAllocator const & ) = delete;
    BumpAllocator( BumpAllocator & ) = delete;
    ~BumpAllocator();

    bool empty() const;
    bool full() const;
    void reset();

    void * allocate( size_t n_bytes );
    uint16_t deallocate( void * );

    bool contains( void * ) const;

private:
    std::atomic< uintptr_t > next_addr;
    uintptr_t const lower_limit;
    uintptr_t const upper_limit;
    std::atomic< uint16_t > count;
};

} // namespace memory
} // namespace redGrapes

