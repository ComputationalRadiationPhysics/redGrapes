/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <unordered_map>
#include <queue>
#include <optional>
#include <functional>
#include <memory>
#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/task/property/graph.hpp>
#include <redGrapes/dispatch/cuda/event_pool.hpp>
#include <redGrapes/dispatch/cuda/task_properties.hpp>

#include <spdlog/spdlog.h>
#include <fmt/format.h>

namespace redGrapes
{
namespace dispatch
{
namespace cuda
{

thread_local cudaStream_t current_stream;

// this class is not thread safe
template <
    typename Task
>
struct CudaStreamDispatcher
{
    cudaStream_t cuda_stream;
    std::recursive_mutex mutex;
    std::queue<
        std::pair<
            cudaEvent_t,
            scheduler::EventPtr
        >
    > events;

    CudaStreamDispatcher()
    {
        cudaStreamCreate( &cuda_stream );
    }

    CudaStreamDispatcher( CudaStreamDispatcher const & other )
    {
        spdlog::warn("CudaStreamDispatcher copy constructor called!");
    }

    ~CudaStreamDispatcher()
    {
        cudaStreamDestroy( cuda_stream );
    }

    void poll()
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        if( ! events.empty() )
        {
            auto & cuda_event = events.front().first;
            auto & event = events.front().second;

            if( cudaEventQuery( cuda_event ) == cudaSuccess )
            {
                SPDLOG_TRACE("cuda event {} ready", cuda_event);
                EventPool::get().free( cuda_event );
                event.notify();

                events.pop();
            }
        }
    }

    void dispatch_task( Task & task )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );

        for(auto predecessor : task.in_edges)
        {
            SPDLOG_TRACE("cudaDispatcher: consider predecessor \"{}\"", predecessor->label);

            if(auto cuda_event = predecessor->cuda_event)
            {
                SPDLOG_TRACE("cudaDispatcher: task {} \"{}\" wait for {}", task.task_id, task.label, *cuda_event);

                cudaStreamWaitEvent( cuda_stream, *cuda_event, 0 );
            }
        }

        SPDLOG_TRACE(
            "CudaScheduler: start {}",
            task_id
        );

        current_stream = cuda_stream;

        // run the code that calls the CUDA API and submits work to current_stream
        task->run();

        cudaEvent_t cuda_event = EventPool::get().alloc();
        cudaEventRecord( cuda_event, cuda_stream );
        task->cuda_event = cuda_event;

        task->get_pre_event().notify();

        SPDLOG_TRACE( "CudaStreamDispatcher {}: recorded event {}", cuda_stream, cuda_event );
        events.push( std::make_pair( cuda_event, task->get_post_event() ) );
    }
};

struct CudaScheduler : redGrapes::scheduler::IScheduler
{
private:
    bool recording;
    bool cuda_graph_enabled;

    std::recursive_mutex mutex;
    unsigned int current_stream;
    std::vector< CudaStreamDispatcher< Task > > streams;

    std::function< bool(Task const&) > is_cuda_task;

public:
    CudaScheduler(
        std::function< bool(Task const&) > is_cuda_task,
        size_t stream_count = 1,
        bool cuda_graph_enabled = false
    ) :
        is_cuda_task( is_cuda_task ),
        current_stream( 0 ),
        cuda_graph_enabled( cuda_graph_enabled )
    {
        // reserve to avoid copy constructor of CudaStreamDispatcher
        streams.reserve( stream_count );

        for( size_t i = 0; i < stream_count; ++i )
            streams.emplace_back();

        SPDLOG_TRACE( "CudaScheduler: use {} streams", streams.size() );
    }

    //! submits the call to the cuda runtime
    void activate_task( Task & task )
    {
        unsigned int stream_id = current_stream;
        current_stream = ( current_stream + 1 ) % streams.size();

        SPDLOG_TRACE( "Dispatch Cuda task {} \"{}\" on stream {}", task.task_id, task.label, stream_id );
        streams[ stream_id ].dispatch_task( task );
    }

    //! checks if some cuda calls finished and notify the redGrapes manager
    void poll()
    {
        for( size_t stream_id = 0; stream_id < streams.size(); ++stream_id )
            streams[ stream_id ].poll();
    }

    /*! whats the task dependency type for the edge a -> b (task a precedes task b)
     * @return true if task b depends on the pre event of task a, false if task b depends on the post event of task b.
     */
    bool task_dependency_type( Task const & a, Task const & b )
    {
        assert( is_cuda_task( b ) );
        return is_cuda_task( a );
    }
};

} // namespace cuda

} // namespace dispatch

} // namespace redGrapes


template <>
struct fmt::formatter< redGrapes::dispatch::cuda::CudaTaskProperties >
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::dispatch::cuda::CudaTaskProperties const & prop,
        FormatContext & ctx
    )
    {
        if( auto e = prop.cuda_event )
            return fmt::format_to( ctx.out(), "\"cuda_event\" : {}", *e );
        else
            return fmt::format_to( ctx.out(), "\"cuda_event\" : null");
    }
};


