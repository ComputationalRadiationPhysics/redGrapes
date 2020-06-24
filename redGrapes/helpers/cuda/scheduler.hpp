/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

//#include <cuda_runtime.h>
#include <map>

#include <redGrapes/helpers/cuda/event_pool.hpp>

namespace redGrapes
{
namespace helpers
{
namespace cuda
{

struct CudaStream
{
    cudaStream_t cuda_stream;

    std::mutex mutex;
    std::queue<
        std::pair<
            cudaEvent_t,
            typename Manager::TaskID
        >
    > events;

    CudaStream()
    {
        cudaStreamCreate( &cuda_stream );
    }

    ~CudaStream()
    {
        cudaStreamDestroy( &cuda_stream );
    }

    void poll()
    {
        std::lock_guard< std::mutex > lock( mutex );

        if( ! events.empty() )
        {
            if( cudaEventQuery( events.front().first ) == cudaSuccess )
            {
                /* TODO
                for( f in followers )
                {
                    if( f.cuda_dependencies[cuda_stream] == f.front().first )
                        f.cuda_dependencies.erase( cuda_stream );
                }
                */

                EventPool::get().free( events.front().first );
                scheduling_graph.task_end( events.front().second );

                events.pop();

                // states[ task_id ] = done;
            }
        }
    }

    void push( TaskPtr & task_ptr )
    {
        auto & task = task_ptr.get();

        /* TODO
        for( auto dep : task.cuda_dependencies )
            cudaStreamWaitEvent( cuda_stream, dep.second );
        */

        // run async cuda call
        (*task.impl)();

        cudaEvent_t cuda_event = EventPool::get().alloc();
        cudaEventRecord( &cuda_event, cuda_stream );

        events.push( std::make_pair( cuda_event, task_id ) );
 
        /* TODO
        for( f in followers )
        {
            if( f is cuda task )
            {
                f.cuda_dependencies[ cuda_stream ] = cuda_event;
            }
        }
        */

        //states[ task_id ] = submitted;
        scheduling_graph.task_start( task_id );
    }
};

struct CudaScheduler
{
public:
    struct TaskProperties
    {
        bool cuda_flag;
    };

    CudaScheduler( size_t stream_count = 1 )
        : streams( stream_count ),
          current_stream( 0 )
    {}

    void activate_task( TaskPtr task_ptr )
    {
        auto & task = task_ptr.locked_get();
        assert( task.is_cuda_task() );

        auto task_id = task.task_id;
        if( scheduling_graph.is_task_ready( task_id ) )
        {
            current_stream = ( current_stream + 1 ) % streams.size();
            streams[ current_stream ].push( task_ptr );
        }
    }

    void poll()
    {
        for( auto & stream : streams )
            stream.poll();
    }

    bool task_dependency_type( TaskPtr a, TaskPtr b )
    {
        assert( b.locked_get().cuda_flag );
        return a.locked_get().cuda_flag;
    }

private:
    std::vector< CudaStream > streams;
    unsigned int current_stream;
};

} // namespace cuda

} // namespace helpers

} // namespace redGrapes


