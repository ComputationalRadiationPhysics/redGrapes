/* Copyright 2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <cstdlib>
#include <hwloc.h>
#include <redGrapes/memory/block.hpp>
#include <spdlog/spdlog.h>

#include <redGrapes/util/trace.hpp>
//#include <redGrapes/redGrapes.hpp>

namespace redGrapes
{

struct HwlocContext
{
    hwloc_topology_t topology;

    HwlocContext()
    {
        hwloc_topology_init(&topology);
        hwloc_topology_load(topology);
    }

    ~HwlocContext()
    {
        hwloc_topology_destroy(topology); 
    }
};

namespace memory
{

struct HwlocAlloc
{
    //! redGrapes context
    HwlocContext & ctx;

    //! hwloc-object used for membind
    hwloc_obj_t obj;

    HwlocAlloc( HwlocContext & ctx, hwloc_obj_t const & obj ) noexcept
        : ctx( ctx ), obj( obj )
    {}

    Block allocate( std::size_t alloc_size ) const noexcept
    {
        TRACE_EVENT("Allocator", "HwlocAlloc::allocate");

        void * ptr = hwloc_alloc_membind(
            ctx.topology, alloc_size, obj->cpuset,
            HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_STRICT
        );

        SPDLOG_TRACE("hwloc_alloc {},{}", (uintptr_t)ptr, alloc_size);

        if( ptr )
            return Block{ (uintptr_t)ptr, alloc_size };
        else
        {
            int error = errno;
            spdlog::error("hwloc_alloc_membind failed: {}\n", strerror(error));
            return Block::null();
        }

        // touch memory
        hwloc_cpuset_t last_cpuset;
        {
            TRACE_EVENT("Allocator", "rebind cpu");
            hwloc_get_cpubind(ctx.topology, last_cpuset, HWLOC_CPUBIND_THREAD);
            hwloc_set_cpubind(ctx.topology, obj->cpuset, HWLOC_CPUBIND_THREAD);
        }

        {
            TRACE_EVENT("Allocator", "memset");
            memset( ptr, 0, alloc_size );
        }

        {
            TRACE_EVENT("Allocator", "rebind cpu");
            hwloc_set_cpubind(ctx.topology, last_cpuset, HWLOC_CPUBIND_THREAD);
        }
    }

    void deallocate( Block blk ) noexcept
    {
        TRACE_EVENT("Allocator", "HwlocAlloc::deallocate");

//        SPDLOG_TRACE("hwloc free {}", (uintptr_t)p);
        hwloc_free( ctx.topology, (void*)blk.ptr, blk.len );
    }
};

} // namespace memory

} // namespace redGrapes

