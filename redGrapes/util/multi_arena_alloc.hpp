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
#include <mutex>
#include <vector>
#include <redGrapes/util/spinlock.hpp>
#include <redGrapes/util/chunked_bump_alloc.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/dispatch/thread/local.hpp>
#include <redGrapes/dispatch/thread/cpuset.hpp>

#include <hwloc.h>

namespace redGrapes
{

extern hwloc_topology_t topology;

namespace memory
{

struct MultiArenaAlloc
{
    std::vector< ChunkedBumpAlloc > arenas;
    
    MultiArenaAlloc( unsigned chunk_size = 0x8000, unsigned n_arenas = 1 )
    {
        size_t n_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
        arenas.reserve(n_pus);

        for(unsigned i = 0; i < n_arenas; ++i)
        {
            hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, (2*i + (2*i)/n_arenas)%n_pus);
            arenas.emplace_back( obj, chunk_size );
        }
    }

    template <typename T>
    T * allocate( unsigned arena_idx, std::size_t n = 1 )
    {
        T * ptr = arenas[arena_idx % arenas.size()].template allocate<T>( n );
        return ptr;
    }

    template <typename T>
    void deallocate( unsigned arena_idx, T * ptr )
    {
        arenas[arena_idx % arenas.size()].template deallocate<T>( ptr );
    }    
};


} // namespace memory
} // namespace redGrapes

