/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>

namespace redGrapes
{
namespace helpers
{
namespace cuda
{

template< typename Manager >
struct StreamSynchronizer
{
    Manager & mgr;
    cudaStream_t cuda_stream;

    std::mutex mutex;
    std::queue< std::pair<cudaEvent_t, typename Manager::EventID> > cuda_events;
    std::vector< cudaEvent_t > unused_cuda_events;

    StreamSynchronizer( Manager & mgr, cudaStream_t cuda_stream ) :
        mgr( mgr ), cuda_stream( cuda_stream )
    {
    }

    ~StreamSynchronizer()
    {
        for( auto e : unused_cuda_events )
            cudaEventDestroy( e );
    }

    void poll()
    {
        std::lock_guard< std::mutex > lock( mutex );

        if( ! cuda_events.empty() )
        {
            auto x = cuda_events.front();
            if( cudaEventQuery( x.first ) == cudaSuccess )
            {
                cuda_events.pop();
                unused_cuda_events.push_back( x.first );
                mgr.reach_event( x.second );
            }
        }
    }

    auto sync()
    {
        return mgr.emplace_task(
            [this]
            {
                std::lock_guard< std::mutex > lock( mutex );

                cudaEvent_t e;
                if( unused_cuda_events.empty() )
                {
                    cudaEventCreate( &e );
                }
                else
                {
                    e = unused_cuda_events.back();
                    unused_cuda_events.pop_back();
                }

                cudaEventRecord( e, cuda_stream );
                cuda_events.push( std::make_pair(e, *mgr.create_event()) );
            }
        );
    }
};

} // namespace cuda

} // namespace helpers
    
} // namespace redGrapes

