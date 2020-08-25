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
    typename TaskPtr
>
struct CuplaStream
{
    cuplaStream_t cupla_stream;
    std::queue<
        std::pair<
            cuplaEvent_t,
            TaskPtr
        >
    > events;

    Cuplastream()
    {
        cuplaStreamCreate( &cupla_stream );
    }

    ~CuplaStream()
    {
        cuplaStreamDestroy( cupla_stream );
    }

    // returns the finished task
    std::optional< TaskPtr > poll()
    {
        if( ! events.empty() )
        {
            auto cupla_event = events.front().first;
            auto task_ptr = events.front().second;

            if( cuplaEventQuery( cupla_event ) == cuplaSuccess )
            {
                EventPool::get().free( cupla_event );
                events.pop();

                return task_ptr;
            }

        }

        return std::nullopt;
    }

    void wait_event( cuplaEvent_t e )
    {
        cuplaStreamWaitEvent( cupla_stream, e, 0 );
    }

    cuplaEvent_t push( TaskPtr & task_ptr )
    {
        // TODO: is there a better way than setting a global variable?
        thread::current_cupla_stream = cupla_stream;

        task_ptr.get().impl->run();

        cuplaEvent_t cupla_event = EventPool::get().alloc();
        cuplaEventRecord( cupla_event, cupla_stream );

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
    typename TaskID,
    typename TaskPtr
>
struct CuplaScheduler : redGrapes::scheduler::SchedulerBase< TaskID, TaskPtr >
{
private:
    bool recording;
    bool cupla_graph_enabled;

    unsigned int current_stream;
    std::vector< CuplaStream< TaskPtr > > streams;

    std::function< bool(TaskPtr) > is_cupla_task;

public:
    CuplaScheduler(
        std::function< bool(TaskPtr) > is_cupla_task,
        size_t stream_count = 8,
        bool cupla_graph_enabled = false
    ) :
        is_cupla_task( is_cupla_task ),
        streams( stream_count ),
        current_stream( 0 ),
        cupla_graph_enabled( cupla_graph_enabled )
    {}

    void activate_task( TaskPtr task_ptr )
    {
        auto task_id = task_ptr.get().task_id;

        if(
            this->scheduling_graph->is_task_ready( task_id ) &&
            ! task_ptr.get().cupla_event
        )
        {
            if( cupla_graph_enabled && ! recording )
            {
                recording = true;
                //TODO: cuplaBeginGraphRecord();

                dispatch_task( task_ptr, task_id );

                //TODO: cuplaEndGraphRecord();
                recording = false;

                //TODO: submitGraph();
            }
            else
                dispatch_task( task_ptr, task_id );
        }
    }

    //! submits the call to the cupla runtime
    void dispatch_task( TaskPtr task_ptr, TaskID task_id )
    {
        current_stream = ( current_stream + 1 ) % streams.size();
        std::cout << "cupla task: " << task_id << " \""<< task_ptr.get().label <<"\", take stream" << current_stream << std::endl;
        for( auto predecessor_ptr : task_ptr.get_predecessors() )
            if( auto cupla_event = predecessor_ptr.get().cupla_event )
            {
                std::cout << "cupla task: " << task_id << " \""<< task_ptr.get().label <<"\", wait for " << *cupla_event << std::endl;
                streams[current_stream].wait_event( *cupla_event );
            }

        this->scheduling_graph->task_start( task_id );
        task_ptr.get().cupla_event = streams[ current_stream ].push( task_ptr );
        std::cout << "cupla task: " << task_id << " \""<< task_ptr.get().label <<"\", event= " << *task_ptr.get().cupla_event << std::endl;

        this->activate_followers( task_ptr );
    }

    //! checks if some cupla calls finished and notify the redGrapes manager
    void poll()
    {
        for( int stream_id = 0; stream_id < streams.size(); ++stream_id )
        {
            if( auto task_ptr = streams[ stream_id ].poll() )
            {
                auto task_id = task_ptr->locked_get().task_id;
                std::cout << "cupla task: " << task_id << " done" << std::endl;

                this->scheduling_graph->task_end( task_id );
                this->activate_followers( *task_ptr );
                this->remove_task( *task_ptr );
            }
        }
    }

    bool task_dependency_type( TaskPtr a, TaskPtr b )
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
    Manager & m,
    std::function< bool(typename Manager::TaskPtr) > is_cupla_task,
    size_t n_streams = 8,
    bool graph_enabled = false
)
{
    return std::make_shared<
               CuplaScheduler<
                   typename Manager::TaskID,
                   typename Manager::TaskPtr
               >
           >(
               is_cupla_task,
               n_streams,
               graph_enabled
           );
}

} // namespace cupla

} // namespace helpers

} // namespace redGrapes


