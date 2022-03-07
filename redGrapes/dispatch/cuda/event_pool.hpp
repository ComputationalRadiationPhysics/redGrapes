/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <vector>
#include <mutex>

namespace redGrapes
{
namespace dispatch
{
namespace cuda
{

//! Manages the recycling of cuda events
struct EventPool
{
public:
    EventPool(EventPool const &) = delete;
    void operator=(EventPool const &) = delete;

    EventPool() {}

    static EventPool & get()
    {
        static EventPool singleton;
        return singleton;
    }

    ~EventPool()
    {
        std::lock_guard< std::mutex > lock( mutex );
        for( auto e : unused_cuda_events )
            cudaEventDestroy( e );
    }

    cudaEvent_t alloc()
    {
        std::lock_guard< std::mutex > lock( mutex );

        cudaEvent_t e;

        if( unused_cuda_events.empty() )
            cudaEventCreate( &e );
        else
        {
            e = unused_cuda_events.back();
            unused_cuda_events.pop_back();
        }

        return e;
    }

    void free( cudaEvent_t event )
    {
        std::lock_guard< std::mutex > lock( mutex );
        unused_cuda_events.push_back( event );
    }

private:
    std::mutex mutex;
    std::vector< cudaEvent_t > unused_cuda_events;

};

} // namespace cuda

} // namespace dispatch
    
} // namespace redGrapes

