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

namespace redGrapes
{
namespace memory
{

struct MultiArenaAlloc
{
    std::vector< std::unique_ptr<ChunkedBumpAlloc> > arenas;
    
    MultiArenaAlloc( unsigned chunk_size = 0x8000, unsigned n_arenas = 1 )
    {
        arenas.reserve(n_arenas);
        for(unsigned i = 0; i < n_arenas; ++i)
        {
            dispatch::thread::pin_cpu( i );
            arenas.push_back( std::make_unique< ChunkedBumpAlloc >(chunk_size) );
        }

        dispatch::thread::unpin_cpu();
    }

    template <typename T>
    T * allocate( unsigned arena_idx, std::size_t n = 1 )
    {
        T * ptr = arenas[arena_idx % arenas.size()]->template allocate<T>( n );
        return ptr;
    }

    template <typename T>
    void deallocate( unsigned arena_idx, T * ptr )
    {
        arenas[arena_idx % arenas.size()]->template deallocate<T>( ptr );
    }    
};


} // namespace memory
} // namespace redGrapes

