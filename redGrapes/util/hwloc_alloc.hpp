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

extern hwloc_topology_t topology;

namespace memory
{

template < typename T >
struct HwlocAlloc
{
    hwloc_obj_t obj;

    typedef T value_type;

    HwlocAlloc( hwloc_obj_t const & obj ) noexcept
        : obj( obj )
    {}

    template< typename U >
    constexpr HwlocAlloc( HwlocAlloc<U> const& other ) noexcept
        : obj( other.obj )
    {}

    T * allocate( std::size_t n ) const
    {
        TRACE_EVENT("Allocator", "HwlocAlloc::allocate");

        size_t alloc_size = sizeof(T) * n;

        SPDLOG_TRACE("hwloc_alloc {} bytes", n);

        void * ptr = hwloc_alloc_membind(
            topology, alloc_size, obj->cpuset,
            HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_NOCPUBIND | HWLOC_MEMBIND_STRICT
        );

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
            TRACE_EVENT("rebind cpu", "Allocator");
            hwloc_get_cpubind(topology, last_cpuset, HWLOC_CPUBIND_THREAD);
            hwloc_set_cpubind(topology, obj->cpuset, HWLOC_CPUBIND_THREAD);
        }

        {
            TRACE_EVENT("memset", "Allocator");
            memset( ptr, 0, alloc_size );
        }

        {
            TRACE_EVENT("rebind cpu", "Allocator");
            hwloc_set_cpubind(topology, last_cpuset, HWLOC_CPUBIND_THREAD);
        }

    }

    void deallocate( T * p, std::size_t n = 0 ) noexcept
    {
        TRACE_EVENT("HwlocAlloc::deallocate", "Allocator");

        SPDLOG_TRACE("hwloc free {}", (uintptr_t)p);
        hwloc_free( topology, (void*)p, sizeof(T)*n );
    }
};

} // namespace memory

} // namespace redGrapes

