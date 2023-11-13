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
#include <redGrapes/memory/block.hpp>
#include <spdlog/spdlog.h>

namespace redGrapes
{
namespace memory
{

/* The `BumpAllocator` manages a chunk of memory,
 * given by `lower_limit` and `upper_limit` by
 * decrementing the `next_addr` by the requested size,
 * and counting the number of active allocations.
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

    Block allocate( size_t n_bytes );

    /*! @return how many active allocations remain,
     * if it returns 0, this allocator needs to be reset.
     */
    uint16_t deallocate( Block );

    bool owns( Block const & ) const;

private:
    std::atomic< uint16_t > count;
    std::atomic< uintptr_t > next_addr;

    uintptr_t const lower_limit;
    uintptr_t const upper_limit;
};

} // namespace memory
} // namespace redGrapes

