/* Copyright 2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <cstdlib>
#include <hwloc.h>
#include <spdlog/spdlog.h>

#include <redGrapes/util/trace.hpp>

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

//extern std::shared_ptr< HwlocContext > hwloc_ctx;

namespace memory
{

template < typename T >
struct HwlocAlloc
{
    std::shared_ptr< HwlocContext > hwloc_ctx;
    hwloc_obj_t obj;

    typedef T value_type;

    HwlocAlloc( std::shared_ptr< HwlocContext > hwloc_ctx, hwloc_obj_t const & obj ) noexcept
        : hwloc_ctx( hwloc_ctx ), obj( obj )
    {}

    template< typename U >
    constexpr HwlocAlloc( HwlocAlloc<U> const& other ) noexcept
        : hwloc_ctx( other.hwloc_ctx ), obj( other.obj )
    {}

    T * allocate( std::size_t n ) const
    {
        TRACE_EVENT("Allocator", "HwlocAlloc::allocate");

        size_t alloc_size = sizeof(T) * n;

        void * ptr = hwloc_alloc_membind(
            hwloc_ctx->topology, alloc_size, obj->cpuset,
            HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_STRICT
        );

        SPDLOG_TRACE("hwloc_alloc {},{}", (uintptr_t)ptr, n);

        if( ptr )
            return (T*)ptr;
        else
        {
            int error = errno;
            spdlog::error("hwloc_alloc_membind failed: {}\n", strerror(error));
            throw std::bad_alloc();
        }

        // touch memory
        hwloc_cpuset_t last_cpuset;
        {
            TRACE_EVENT("Allocator", "rebind cpu");
            hwloc_get_cpubind(hwloc_ctx->topology, last_cpuset, HWLOC_CPUBIND_THREAD);
            hwloc_set_cpubind(hwloc_ctx->topology, obj->cpuset, HWLOC_CPUBIND_THREAD);
        }

        {
            TRACE_EVENT("Allocator", "memset");
            memset( ptr, 0, alloc_size );
        }

        {
            TRACE_EVENT("Allocator", "rebind cpu");
            hwloc_set_cpubind(hwloc_ctx->topology, last_cpuset, HWLOC_CPUBIND_THREAD);
        }

    }

    void deallocate( T * p, std::size_t n = 0 ) noexcept
    {
        TRACE_EVENT("Allocator", "HwlocAlloc::deallocate");

//        SPDLOG_TRACE("hwloc free {}", (uintptr_t)p);
        hwloc_free( hwloc_ctx->topology, (void*)p, sizeof(T)*n );
    }
};

} // namespace memory

} // namespace redGrapes

