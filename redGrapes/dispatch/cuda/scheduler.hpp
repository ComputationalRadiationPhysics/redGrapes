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
#include <redGrapes/scheduler/scheduling_graph.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/dispatch/cuda/event_pool.hpp>

#include <spdlog/spdlog.h>
#include <fmt/format.h>

namespace redGrapes
{
namespace dispatch
{

namespace thread
{
thread_local cudaStream_t current_cuda_stream;
}

namespace cuda
{

// this class is not thread safe
template <
    typename Task
>
struct CudaStream
{
    cudaStream_t cuda_stream;
    std::recursive_mutex mutex;
    std::queue<
        std::pair<
            cudaEvent_t,
            typename Task::VertexPtr
        >
    > events;

    CudaStream()
    {
        cudaStreamCreate( &cuda_stream );
    }

    CudaStream( CudaStream const & other )
    {
        spdlog::warn("CudaStream copy constructor called!");
    }

    ~CudaStream()
    {
        cudaStreamDestroy( cuda_stream );
    }

    // returns the finished task
    std::optional< typename Task::VertexPtr > poll()
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        if( ! events.empty() )
        {
            auto cuda_event = events.front().first;
            auto task_ptr = events.front().second;

            if( cudaEventQuery( cuda_event ) == cudaSuccess )
            {
                SPDLOG_TRACE("cuda event {} ready", cuda_event);
                EventPool::get().free( cuda_event );
                events.pop();

                return task_ptr;
            }
        }

        return std::nullopt;
    }

    void wait_event( cudaEvent_t e )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        cudaStreamWaitEvent( cuda_stream, e, 0 );
    }

    cudaEvent_t push( typename Task::VertexPtr task_ptr )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );

        // TODO: is there a better way than setting a global variable?
        thread::current_cuda_stream = cuda_stream;

        task_ptr->task->impl->run();

        cudaEvent_t cuda_event = EventPool::get().alloc();
        cudaEventRecord( cuda_event, cuda_stream );
        task_ptr->task->cuda_event = cuda_event;

        auto pe = task_ptr->task->pre_event;
        pe->reach();

        SPDLOG_TRACE( "CudaStream {}: recorded event {}", cuda_stream, cuda_event );
        events.push( std::make_pair( cuda_event, task_ptr ) );

        return cuda_event;
    }
};

struct CudaTaskProperties
{
    std::optional< cudaEvent_t > cuda_event;

    CudaTaskProperties() {}

    template < typename PropertiesBuilder >
    struct Builder
    {
        PropertiesBuilder & builder;

        Builder( PropertiesBuilder & b )
            : builder(b)
        {}
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
    typename Task
>
struct CudaScheduler : redGrapes::scheduler::IScheduler< Task >
{
private:
    bool recording;
    bool cuda_graph_enabled;

    std::recursive_mutex mutex;
    unsigned int current_stream;
    std::vector< CudaStream< Task > > streams;

    std::function< bool(typename Task::VertexPtr) > is_cuda_task;

    IManager< Task > & mgr;

public:
    CudaScheduler(
        IManager<Task> & mgr,
        std::function< bool(typename Task::VertexPtr) > is_cuda_task,
        size_t stream_count = 1,
        bool cuda_graph_enabled = false
    ) :
        mgr(mgr),
        is_cuda_task( is_cuda_task ),
        current_stream( 0 ),
        cuda_graph_enabled( cuda_graph_enabled )
    {
        // reserve to avoid copy constructor of CudaStream
        streams.reserve( stream_count );

        for( size_t i = 0; i < stream_count; ++i )
            streams.emplace_back();

        SPDLOG_TRACE( "CudaScheduler: use {} streams", streams.size() );
    }

    bool activate_task( typename Task::VertexPtr task_ptr )
    {
        auto task_id = task_ptr->task->task_id;
        SPDLOG_TRACE("CudaScheduler: activate task {} \"{}\"", task_id, task_ptr->task->label);

        if( task_ptr->task->is_ready() )
        {
            if(!task_ptr->task->in_ready_list.test_and_set())
            {
                std::unique_lock< std::recursive_mutex > lock( mutex );

                if( cuda_graph_enabled && ! recording )
                {
                    recording = true;
                    //TODO: cudaBeginGraphRecord();

                    dispatch_task( lock, task_ptr, task_id );

                    //TODO: cudaEndGraphRecord();
                    recording = false;

                    //TODO: submitGraph();
                }
                else
                    dispatch_task( lock, task_ptr, task_id );

                mgr.get_scheduler()->notify();
                
                return true;
            }
        }

        return false;
    }

    //! submits the call to the cuda runtime
    void dispatch_task( std::unique_lock< std::recursive_mutex > & lock, typename Task::VertexPtr task_ptr, TaskID task_id )
    {
        unsigned int stream_id = current_stream;
        current_stream = ( current_stream + 1 ) % streams.size();

        SPDLOG_TRACE( "Dispatch Cuda task {} \"{}\" on stream {}", task_id, task_ptr->task->label, stream_id );

        for(auto weak_predecessor_ptr : task_ptr->in_edges)
        {
            if(auto predecessor_ptr = weak_predecessor_ptr.lock())
            {
                SPDLOG_TRACE("cuda scheduler: consider predecessor \"{}\"", predecessor_ptr->task->label);

                if(auto cuda_event = predecessor_ptr->task->cuda_event)
                {
                    SPDLOG_TRACE("cuda task {} \"{}\" wait for {}", task_id, task_ptr->task->label, *cuda_event);

                    streams[stream_id].wait_event(*cuda_event);
                }
            }
        }

        SPDLOG_TRACE(
            "CudaScheduler: start {}",
            task_id
        );

        streams[ stream_id ].push( task_ptr );

        lock.unlock();
        
        SPDLOG_TRACE(
            "CudaScheduler: task {} \"{}\"::event = {}",
            task_id,
            task_ptr->task->label,
            *task_ptr->task->cuda_event
        );
    }

    //! checks if some cuda calls finished and notify the redGrapes manager
    void poll()
    {
        for( size_t stream_id = 0; stream_id < streams.size(); ++stream_id )
        {
            if( auto task_ptr = streams[ stream_id ].poll() )
            {
                auto task_id = (*task_ptr)->task->task_id;
                SPDLOG_TRACE( "cuda task {} done", task_id );

                auto pe = (*task_ptr)->task->post_event;
                pe->reach();

                mgr.get_scheduler()->notify();
            }
        }
    }

    /*! whats the task dependency type for the edge a -> b (task a precedes task b)
     * @return true if task b depends on the pre event of task a, false if task b depends on the post event of task b.
     */
    bool task_dependency_type( typename Task::VertexPtr a, typename Task::VertexPtr b )
    {
        assert( is_cuda_task( b ) );
        return is_cuda_task( a );
    }
};


/*! Factory function to easily create a cuda-scheduler object
 */
template <
    typename Manager
>
auto make_cuda_scheduler(
    Manager & mgr,
    std::function< bool(typename Manager::Task::VertexPtr) > is_cuda_task,
    size_t n_streams = 8,
    bool graph_enabled = false
)
{
    return std::make_shared<
        CudaScheduler< typename Manager::Task >
           >(
               mgr,
               is_cuda_task,
               n_streams,
               graph_enabled
           );
}

} // namespace cuda

} // namespace dispatch

} // namespace redGrapes


template <>
struct fmt::formatter< redGrapes::helpers::cuda::CudaTaskProperties >
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::helpers::cuda::CudaTaskProperties const & prop,
        FormatContext & ctx
    )
    {
        if( auto e = prop.cuda_event )
            return fmt::format_to( ctx.out(), "\"cuda_event\" : {}", *e );
        else
            return fmt::format_to( ctx.out(), "\"cuda_event\" : null");
    }
};


