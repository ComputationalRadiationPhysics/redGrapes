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
#include <redGrapes/dispatch/cupla/event_pool.hpp>
#include <redGrapes/dispatch/cupla/task_properties.hpp>

#include <spdlog/spdlog.h>
#include <fmt/format.h>

namespace redGrapes
{
namespace dispatch
{
namespace cupla
{

thread_local cuplaStream_t current_stream;

// this class is not thread safe
template <
    typename Task
>
struct CuplaStreamDispatcher
{
    cuplaStream_t cupla_stream;
    std::recursive_mutex mutex;
    std::queue<
        std::pair<
            cuplaEvent_t,
            scheduler::EventPtr
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

    void poll()
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        if( ! events.empty() )
        {
            auto & cupla_event = events.front().first;
            auto & event = events.front().second;

            if( cuplaEventQuery( cupla_event ) == cuplaSuccess )
            {
                SPDLOG_TRACE("cupla event {} ready", cupla_event);
                EventPool::get().free( cupla_event );
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
            SPDLOG_TRACE("cuplaDispatcher: consider predecessor \"{}\"", predecessor->label);

            if(auto cupla_event = predecessor->cupla_event)
            {
                SPDLOG_TRACE("cuplaDispatcher: task {} \"{}\" wait for {}", task.task_id, task.label, *cupla_event);

                cuplaStreamWaitEvent( cupla_stream, *cupla_event, 0 );
            }
        }

        SPDLOG_TRACE(
            "CuplaScheduler: start {}",
            task.task_id
        );

        current_stream = cupla_stream;

        // run the code that calls the CUDA API and submits work to current_stream
        task->run();

        cuplaEvent_t cupla_event = EventPool::get().alloc();
        cuplaEventRecord( cupla_event, cupla_stream );
        task->cupla_event = cupla_event;

        task->get_pre_event().notify();

        SPDLOG_TRACE( "CuplaStreamDispatcher {}: recorded event {}", cupla_stream, cupla_event );
        events.push( std::make_pair( cupla_event, task->get_post_event() ) );
    }
};

struct CuplaScheduler : redGrapes::scheduler::IScheduler
{
private:
    bool recording;
    bool cupla_graph_enabled;

    std::recursive_mutex mutex;
    unsigned int current_stream;
    std::vector< CuplaStreamDispatcher< Task > > streams;

    std::function< bool(Task const&) > is_cupla_task;

public:
    CuplaScheduler(
        std::function< bool(Task const&) > is_cupla_task,
        size_t stream_count = 1,
        bool cupla_graph_enabled = false
    ) :
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

    //! submits the call to the cupla runtime
    void activate_task( Task & task )
    {
        unsigned int stream_id = current_stream;
        current_stream = ( current_stream + 1 ) % streams.size();

        SPDLOG_TRACE( "Dispatch Cupla task {} \"{}\" on stream {}", task.task_id, task.label, stream_id );
        streams[ stream_id ].dispatch_task( task );
    }

    //! checks if some cupla calls finished and notify the redGrapes manager
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
        assert( is_cupla_task( b ) );
        return is_cupla_task( a );
    }
};

} // namespace cupla

} // namespace dispatch

} // namespace redGrapes


template <>
struct fmt::formatter< redGrapes::dispatch::cupla::CuplaTaskProperties >
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::dispatch::cupla::CuplaTaskProperties const & prop,
        FormatContext & ctx
    )
    {
        if( auto e = prop.cupla_event )
            return fmt::format_to( ctx.out(), "\"cupla_event\" : {}", *e );
        else
            return fmt::format_to( ctx.out(), "\"cupla_event\" : null");
    }
};


