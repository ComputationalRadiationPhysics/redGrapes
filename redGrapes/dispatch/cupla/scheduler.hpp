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
#include <redGrapes/dispatch/cupla/event_pool.hpp>

#include <spdlog/spdlog.h>
#include <fmt/format.h>

namespace redGrapes
{
namespace dispatch
{

namespace thread
{
thread_local cuplaStream_t current_cupla_stream;
}

namespace cupla
{

// this class is not thread safe
template <
    typename Task
>
struct CuplaStreamDispatcher : IDispatcher
{
    cuplaStream_t cupla_stream;
    std::recursive_mutex mutex;
    std::queue<
        std::pair<
            cuplaEvent_t,
            typename Task::VertexPtr
        >
    > events;

    CuplaStreamDispatcher()
    {
        cuplaStreamCreate( &cupla_stream );
    }

    CuplaStreamDispatcher( CuplaStreamDispatcher const & other )
    {
        spdlog::warn("CuplaStreamDispatcher copy constructor called!");
    }

    ~CuplaStreamDispatcher()
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

    void dispatch_task( typename Task::VertexPtr task_vertex )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );

        for(auto weak_predecessor_ptr : task_vertex->in_edges)
        {
            if(auto predecessor_ptr = weak_predecessor_ptr.lock())
            {
                SPDLOG_TRACE("cuplaDispatcher: consider predecessor \"{}\"", predecessor_ptr->task->label);

                if(auto cupla_event = predecessor_ptr->task->cupla_event)
                {
                    SPDLOG_TRACE("cuplaDispatcher: task {} \"{}\" wait for {}", task_id, task_vertex->task->label, *cupla_event);

                    cuplaStreamWaitEvent( cupla_stream, *cupla_event, 0 );
                }
            }
        }

        SPDLOG_TRACE(
            "CuplaScheduler: start {}",
            task_id
        );

        // TODO: is there a better way than setting a global variable?
        thread::current_cupla_stream = cupla_stream;

        task_vertex->task->impl->run();

        cuplaEvent_t cupla_event = EventPool::get().alloc();
        cuplaEventRecord( cupla_event, cupla_stream );
        task_vertex->task->cupla_event = cupla_event;

        auto pe = task_vertex->task->pre_event;
        pe->reach();

        SPDLOG_TRACE( "CuplaStreamDispatcher {}: recorded event {}", cupla_stream, cupla_event );
        events.push( std::make_pair( cupla_event, task_vertex ) );

        SPDLOG_TRACE(
                     "CuplaScheduler: task {} \"{}\"::event = {}",
                     task_id,
                     task_vertex->task->label,
                     *task_vertex->task->cupla_event
                     );
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
    std::vector< CuplaStreamDispatcher< Task > > streams;

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
        // reserve to avoid copy constructor of CuplaStreamDispatcher
        streams.reserve( stream_count );

        for( size_t i = 0; i < stream_count; ++i )
            streams.emplace_back();

        SPDLOG_TRACE( "CuplaScheduler: use {} streams", streams.size() );
    }

    bool notify_event( int status, std::shared_ptr<Event> event )
    {
        if( status == 1 )
        {
        auto task_id = task_vertex->task->task_id;
        SPDLOG_TRACE("CuplaScheduler: activate task {} \"{}\"", task_id, task_vertex->task->label);

        if( task_vertex->task->is_ready() )
        {
            if(!task_vertex->task->in_ready_list.test_and_set())
            {
                std::unique_lock< std::recursive_mutex > lock( mutex );

                if( cupla_graph_enabled && ! recording )
                {
                    recording = true;
                    //TODO: cuplaBeginGraphRecord();

                    dispatch_task( task_vertex, task_id );

                    //TODO: cuplaEndGraphRecord();
                    recording = false;

                    //TODO: submitGraph();
                }
                else
                    dispatch_task( task_vertex, task_id );

                mgr.get_scheduler()->notify();
                
                return true;
            }
        }

        return false;
    }

    //! submits the call to the cupla runtime
    void dispatch_task( TaskVertexPtr task_vertex )
    {
        unsigned int stream_id = current_stream;
        current_stream = ( current_stream + 1 ) % streams.size();

        SPDLOG_TRACE( "Dispatch Cupla task {} \"{}\" on stream {}", task_vertex->task->id, task_vertex->task->label, stream_id );
        streams[ stream_id ].dispatch_task( task_ptr );
    }

    //! checks if some cupla calls finished and notify the redGrapes manager
    void poll()
    {
        for( size_t stream_id = 0; stream_id < streams.size(); ++stream_id )
        {
            if( auto task_ptr = streams[ stream_id ].poll() )
            {
                auto task = (*task_ptr)->task;
                SPDLOG_TRACE( "cupla task {} done", task.task_id );

                mgr.notify_event( task.post_event );
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

} // namespace dispatch

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


