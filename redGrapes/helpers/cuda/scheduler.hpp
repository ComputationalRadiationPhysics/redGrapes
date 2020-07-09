/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

//#include <cuda_runtime.h>
#include <unordered_map>

#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/helpers/cuda/event_pool.hpp>

namespace redGrapes
{

namespace thread
{
thread_local cudaStream_t current_cuda_stream;
}

namespace helpers
{
namespace cuda
{

// this class is not thread safe
template <
    typename TaskPtr
>
struct CudaStream
{
    cudaStream_t cuda_stream;
    std::queue<
        std::pair<
            cudaEvent_t,
            TaskPtr
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

    // returns the finished task
    std::optional< TaskPtr > poll()
    {
        if( ! events.empty() )
        {
            auto cuda_event = events.front().first;
            auto task_ptr = events.front().second;

            if( cudaEventQuery( cuda_event ) == cudaSuccess )
            {
                EventPool::get().free( cuda_event );
                events.pop();

                return task_ptr;
            }
        }

        return std::nullopt;
    }

    void wait_event( cudaEvent_t e )
    {
        cudaStreamWaitEvent( cuda_stream, e );        
    }

    cudaEvent_t push( TaskPtr & task_ptr )
    {
        scheduling_graph.begin_task( task_ptr.locked_get().task_id );

        // todo: is there a better way than setting a global variable?
        thread::current_cuda_stream = cuda_stream;
        mgr_run_task( task_ptr );

        cudaEvent_t cuda_event = EventPool::get().alloc();
        cudaEventRecord( &cuda_event, cuda_stream );

        events.push( std::make_pair( cuda_event, task_ptr ) );

        return cuda_event;
    }
};

struct CudaScheduler : IScheduler
{
private:
    //! todo: manager interface
    SchedulingGraph< TaskID, TaskPtr > & scheduling_graph;
    std::function< bool ( TaskPtr ) > mgr_run_task;
    std::function< void ( TaskPtr ) > mgr_activate_followers;
    std::function< void ( TaskPtr ) > mgr_remove_task;


    bool recording;
    bool cuda_graph_enabled;

    unsigned int current_stream;
    std::vector< CudaStream > streams;

    std::unordered_map<
        TaskID,
        std::vector< std::optional< cudaEvent_t > >
    > cuda_dependencies;

public:
    struct TaskProperties
    {
        bool cuda_flag;
    };

    CudaScheduler(
        SchedulingGraph<TaskID, TaskPtr> & scheduling_graph,
        std::function< bool ( TaskPtr ) > mgr_run_task,
        std::function< void ( TaskPtr ) > mgr_activate_followers,
        std::function< void ( TaskPtr ) > mgr_remove_task,

        size_t stream_count = 1
    ) :
        scheduling_graph( scheduling_graph ),
        mgr_run_task( mgr_run_task ),
        mgr_activate_followers( mgr_activate_followers ),
        mgr_remove_task( mgr_remove_task ),

        streams( stream_count ),
        current_stream( 0 ),
        cuda_graph_enabled( false )
    {}

    void activate_task( TaskPtr task_ptr )
    {
        auto task_id = task_ptr.locked_get().task_id;

        if( scheduling_graph.is_task_ready( task_id ) )
        {
            if( cuda_graph_enabled && ! recording )
            {
                recording = true;
                //TODO: cudaBeginGraphRecord();

                dispatch_task( task_ptr );

                //TODO: cudaEndGraphRecord();
                recording = false;

                //TODO: submitGraph();
            }
            else
                dispatch_task( task_ptr );
        }
    }

    //! submits the call to the cuda runtime
    void dispatch_task( TaskPtr task_ptr )
    {
        current_stream = ( current_stream + 1 ) % streams.size();

        assert( cuda_dependencies.count( task_id ) );        
        for( int stream_id = 0; stream_id < streams.size(); ++stream_id )
            if( auto cuda_event = cuda_dependencies[ task_id ][ stream_id ] )
                streams[ current_stream ].wait_event( *cuda_event );

        cudaEvent_t cuda_event = streams[ current_stream ].push( task_ptr );
        /* TODO
        for( f in task.followers )
        {
            if( task_ptr.get().cuda_flag )
            {
                cuda_dependencies[ f ].resize( streams.size() );
                cuda_dependencies[ f ][ current_stream ] = cuda_event;
            }
        }
        */
        mgr_activate_followers( task_id );
    }

    //! checks if some cuda calls finished and notify the redGrapes manager
    void poll()
    {
        for( int stream_id = 0; i < streams.size(); ++i )
        {
            if( auto task_id = streams[ stream_id ].poll() )
            {
                /* TODO
                for( f in followers( task_id ) )
                {
                    if( cuda_dependencies.count( f ) )
                    {
                        assert( cuda_dependencies[ f ].size() == streams.size() );
                        cuda_dependencies[ f ][ stream_id ] = std::nullopt;
                    }
                }
                */

                scheduling_graph.task_end( task_id );
                mgr_activate_followers( task_ptr );
                mgr_remove_task( task_ptr );
            }
        }
    }

    bool task_dependency_type( TaskPtr a, TaskPtr b )
    {
        assert( b.get().cuda_flag );
        return a.get().cuda_flag;
    }
};

} // namespace cuda

} // namespace helpers

} // namespace redGrapes


