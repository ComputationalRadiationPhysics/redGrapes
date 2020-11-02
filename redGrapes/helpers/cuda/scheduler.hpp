/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <queue>
#include <optional>
#include <functional>
#include <memory>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/graph/scheduling_graph.hpp>
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
        cudaStreamDestroy( cuda_stream );
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
        cudaStreamWaitEvent( cuda_stream, e, 0 );
    }

    cudaEvent_t push( TaskPtr & task_ptr )
    {
        // TODO: is there a better way than setting a global variable?
        thread::current_cuda_stream = cuda_stream;

        task_ptr.get().impl->run();

        cudaEvent_t cuda_event = EventPool::get().alloc();
        cudaEventRecord( cuda_event, cuda_stream );

        events.push( std::make_pair( cuda_event, task_ptr ) );

        return cuda_event;
    }
};

struct CudaTaskProperties
{
    bool cuda_flag;
    std::optional< cudaEvent_t > cuda_event;

    CudaTaskProperties()
        : cuda_flag( false )
    {}

    template < typename PropertiesBuilder >
    struct Builder
    {
        PropertiesBuilder & builder;

        Builder( PropertiesBuilder & b )
            : builder(b)
        {}

        PropertiesBuilder cuda_task()
        {
            builder.prop.cuda_flag = true;
            return builder;
        }
    };

    struct Patch
    {
        template <typename PatchBuilder>
        struct Builder
        {
            Builder( PatchBuilder & ) {}
        };
    };
    void apply_patch( Patch const & ) {};
};

template <
    typename TaskID,
    typename TaskPtr
>
struct CudaScheduler : redGrapes::scheduler::SchedulerBase< TaskID, TaskPtr >
{
private:
    bool recording;
    bool cuda_graph_enabled;

    unsigned int current_stream;
    std::vector< CudaStream< TaskPtr > > streams;

public:
    CudaScheduler(
        size_t stream_count = 8,
        bool cuda_graph_enabled = false
    ) :
        streams( stream_count ),
        current_stream( 0 ),
        cuda_graph_enabled( cuda_graph_enabled )
    {}

    void activate_task( TaskPtr task_ptr )
    {
        auto task_id = task_ptr.get().task_id;

        if(
            this->scheduling_graph->is_task_ready( task_id ) &&
            ! task_ptr.get().cuda_event
        )
        {
            if( cuda_graph_enabled && ! recording )
            {
                recording = true;
                //TODO: cudaBeginGraphRecord();

                dispatch_task( task_ptr, task_id );

                //TODO: cudaEndGraphRecord();
                recording = false;

                //TODO: submitGraph();
            }
            else
                dispatch_task( task_ptr, task_id );
        }
    }

    //! submits the call to the cuda runtime
    void dispatch_task( TaskPtr task_ptr, TaskID task_id )
    {
        current_stream = ( current_stream + 1 ) % streams.size();

        for( auto predecessor_ptr : task_ptr.get_predecessors() )
            if( auto cuda_event = predecessor_ptr.get().cuda_event )
                streams[current_stream].wait_event( *cuda_event );

        this->scheduling_graph->task_start( task_id );
        task_ptr.get().cuda_event = streams[ current_stream ].push( task_ptr );

        this->activate_followers( task_ptr );
    }

    //! checks if some cuda calls finished and notify the redGrapes manager
    void poll()
    {
        for( int stream_id = 0; stream_id < streams.size(); ++stream_id )
        {
            if( auto task_ptr = streams[ stream_id ].poll() )
            {
                auto task_id = task_ptr->locked_get().task_id;

                this->scheduling_graph->task_end( task_id );
                this->activate_followers( *task_ptr );
                this->remove_task( *task_ptr );
            }
        }
    }

    bool task_dependency_type( TaskPtr a, TaskPtr b )
    {
        assert( b.get().cuda_flag );
        return a.get().cuda_flag;
    }
};


/*! Factory function to easily create a cuda-scheduler object
 */
template <
    typename Manager
>
auto make_cuda_scheduler(
    Manager & m,
    size_t n_streams = 8,
    bool graph_enabled = false
)
{
    return std::make_shared<
               CudaScheduler<
                   typename Manager::TaskID,
                   typename Manager::TaskPtr
               >
           >(
               n_streams,
               graph_enabled
           );
}

} // namespace cuda

} // namespace helpers

} // namespace redGrapes


