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
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/graph/scheduling_graph.hpp>
#include <redGrapes/helpers/cupla/event_pool.hpp>

#include <spdlog/spdlog.h>
#include <fmt/format.h>

namespace redGrapes
{

namespace thread
{
thread_local cuplaStream_t current_cupla_stream;
}

namespace helpers
{
namespace cupla
{

// this class is not thread safe
template <
    typename Task
>
struct CuplaStream
{
    cuplaStream_t cupla_stream;
    std::recursive_mutex mutex;
    std::queue<
        std::pair<
            cuplaEvent_t,
            typename Task::VertexPtr
        >
    > events;

    CuplaStream()
    {
        cuplaStreamCreate( &cupla_stream );
    }

    CuplaStream( CuplaStream const & other )
    {
        spdlog::warn("CuplaStream copy constructor called!");
    }

    ~CuplaStream()
    {
        cuplaStreamDestroy( cupla_stream );
    }

    // returns the finished task
    std::optional< typename Task::VertexPtr > poll()
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        if( ! events.empty() )
        {
            auto cupla_event = events.front().first;
            auto task_ptr = events.front().second;

            if( cuplaEventQuery( cupla_event ) == cuplaSuccess )
            {
                SPDLOG_TRACE("cupla event {} ready", cupla_event);
                EventPool::get().free( cupla_event );
                events.pop();

                return task_ptr;
            }
        }

        return std::nullopt;
    }

    void wait_event( cuplaEvent_t e )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        cuplaStreamWaitEvent( cupla_stream, e, 0 );
    }

    cuplaEvent_t push( typename Task::VertexPtr task_ptr, SchedulingGraph<Task> & sg )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );

        // TODO: is there a better way than setting a global variable?
        thread::current_cupla_stream = cupla_stream;

        task_ptr->task->impl->run();

        cuplaEvent_t cupla_event = EventPool::get().alloc();
        cuplaEventRecord( cupla_event, cupla_stream );

        task_ptr->task->cupla_event = cupla_event;
        sg.task_start( task_ptr->task->task_id );

        SPDLOG_TRACE( "CuplaStream {}: recorded event {}", cupla_stream, cupla_event );
        events.push( std::make_pair( cupla_event, task_ptr ) );

        return cupla_event;
    }
};

struct CuplaTaskProperties
{
    std::optional< cuplaEvent_t > cupla_event;

    CuplaTaskProperties() {}

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
struct CuplaScheduler : redGrapes::scheduler::IScheduler< Task >
{
private:
    bool recording;
    bool cupla_graph_enabled;

    std::recursive_mutex mutex;
    unsigned int current_stream;
    std::vector< CuplaStream< Task > > streams;

    std::function< bool(typename Task::VertexPtr) > is_cupla_task;

    IManager< Task > & mgr;

public:
    CuplaScheduler(
                   IManager<Task> & mgr,
        std::function< bool(typename Task::VertexPtr) > is_cupla_task,
        size_t stream_count = 1,
        bool cupla_graph_enabled = false
    ) :
        mgr(mgr),
        is_cupla_task( is_cupla_task ),
        current_stream( 0 ),
        cupla_graph_enabled( cupla_graph_enabled )
    {
        // reserve to avoid copy constructor of CuplaStream
        streams.reserve( stream_count );

        for( size_t i = 0; i < stream_count; ++i )
            streams.emplace_back();

        SPDLOG_TRACE( "CuplaScheduler: use {} streams", streams.size() );
    }

    bool activate_task( typename Task::VertexPtr task_ptr )
    {
        auto task_id = task_ptr->task->task_id;
        SPDLOG_TRACE("CuplaScheduler: activate task {} \"{}\"", task_id, task_ptr->task->label);

        if(mgr.get_scheduling_graph()->is_task_ready( task_id ) )
        {
            if(!task_ptr->task->in_ready_list.test_and_set())
            {
                std::unique_lock< std::recursive_mutex > lock( mutex );

                if( cupla_graph_enabled && ! recording )
                {
                    recording = true;
                    //TODO: cuplaBeginGraphRecord();

                    dispatch_task( lock, task_ptr, task_id );

                    //TODO: cuplaEndGraphRecord();
                    recording = false;

                    //TODO: submitGraph();
                }
                else
                    dispatch_task( lock, task_ptr, task_id );

                return true;
            }
        }

        return false;
    }

    //! submits the call to the cupla runtime
    void dispatch_task( std::unique_lock< std::recursive_mutex > & lock, typename Task::VertexPtr task_ptr, TaskID task_id )
    {
        unsigned int stream_id = current_stream;
        current_stream = ( current_stream + 1 ) % streams.size();

        SPDLOG_TRACE( "Dispatch Cupla task {} \"{}\" on stream {}", task_id, task_ptr->task->label, stream_id );

        for(auto weak_predecessor_ptr : task_ptr->in_edges)
        {
            if(auto predecessor_ptr = weak_predecessor_ptr.lock())
            {
                SPDLOG_TRACE("cupla scheduler: consider predecessor \"{}\"", predecessor_ptr->task->label);

                if(auto cupla_event = predecessor_ptr->task->cupla_event)
                {
                    SPDLOG_TRACE("cupla task {} \"{}\" wait for {}", task_id, task_ptr->task->label, *cupla_event);

                    streams[stream_id].wait_event(*cupla_event);
                }
            }
        }

        SPDLOG_TRACE(
            "CuplaScheduler: start {}",
            task_id
        );

        streams[ stream_id ].push( task_ptr, *mgr.get_scheduling_graph() );

        lock.unlock();
        
        SPDLOG_TRACE(
            "CuplaScheduler: task {} \"{}\"::event = {}",
            task_id,
            task_ptr->task->label,
            *task_ptr->task->cupla_event
        );
    }

    //! checks if some cupla calls finished and notify the redGrapes manager
    void poll()
    {
        for( size_t stream_id = 0; stream_id < streams.size(); ++stream_id )
        {
            if( auto task_ptr = streams[ stream_id ].poll() )
            {
                auto task_id = (*task_ptr)->task->task_id;
                SPDLOG_TRACE( "cupla task {} done", task_id );

                mgr.get_scheduling_graph()->task_end( task_id );
                mgr.remove_task( *task_ptr );
            }
        }
    }

    /*! whats the task dependency type for the edge a -> b (task a precedes task b)
     * @return true if task b depends on the pre event of task a, false if task b depends on the post event of task b.
     */
    bool task_dependency_type( typename Task::VertexPtr a, typename Task::VertexPtr b )
    {
        assert( is_cupla_task( b ) );
        return is_cupla_task( a );
    }
};


/*! Factory function to easily create a cupla-scheduler object
 */
template <
    typename Manager
>
auto make_cupla_scheduler(
    Manager & mgr,
    std::function< bool(typename Manager::Task::VertexPtr) > is_cupla_task,
    size_t n_streams = 8,
    bool graph_enabled = false
)
{
    return std::make_shared<
        CuplaScheduler< typename Manager::Task >
           >(
               mgr,
               is_cupla_task,
               n_streams,
               graph_enabled
           );
}

} // namespace cupla

} // namespace helpers

} // namespace redGrapes


template <>
struct fmt::formatter< redGrapes::helpers::cupla::CuplaTaskProperties >
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::helpers::cupla::CuplaTaskProperties const & prop,
        FormatContext & ctx
    )
    {
        if( auto e = prop.cupla_event )
            return fmt::format_to( ctx.out(), "\"cupla_event\" : {}", *e );
        else
            return fmt::format_to( ctx.out(), "\"cupla_event\" : null");
    }
};


