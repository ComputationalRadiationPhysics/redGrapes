/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <cassert>
#include <condition_variable>
#include <boost/graph/adjacency_list.hpp>
#include <redGrapes/graph/recursive_graph.hpp>
#include <redGrapes/graph/precedence_graph.hpp>
#include <redGrapes/graph/util.hpp>
#include <redGrapes/task/task.hpp>

#include <redGrapes/thread/thread_schedule.hpp>
#include <redGrapes/thread/thread_dispatcher.hpp>

namespace redGrapes
{

template <
    typename TaskID,
    typename TaskPtr
>
class SchedulingGraph
{
public:
    using EventID = unsigned int;

private:
    struct Event
    {
        // number of incoming edges
        // state == 0: event is reached and can be removed
        unsigned int state;
        std::vector< EventID > followers;

        Event()
            // every event needs at least one down() before
            // it will be removed
            : state( 1 )
        {}

        bool is_reached()
        {
            return state == 0;
        }

        void up()
        {
            state += 1;
        }

        void down()
        {
            state -= 1;
        }
    };

    struct TaskEvents
    {
        TaskID pre_event;
        TaskID post_event;
    };

    std::mutex mutex;
    std::unordered_map< EventID, Event > events;
    std::unordered_map< TaskID , TaskEvents > task_events;

    /*!
     * Create a new event (with no dependencies)
     *
     * Not thread safe!
     */
    EventID make_event()
    {
        static EventID event_id_counter = 0;

        EventID event_id = event_id_counter ++;

        events.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(event_id),
            std::forward_as_tuple()
        );

        return event_id;
    }

    /*!
     * Event a precedes event b
     *
     * Not thread safe!
     */
    void add_edge( EventID a, EventID b )
    {
        assert( events.count( a ) );
        assert( events.count( b ) );

        events[ a ].followers.push_back( b );
        events[ b ].up();
    }

    /*
     * not thread safe
     */
    void remove_edge( EventID a, EventID b )
    {
        assert( events.count( a ) );
        assert( events.count( b ) );

        auto & fs = events[ a ].followers;
        fs.erase(
            std::find( std::begin(fs), std::end(fs), b)
        );
        events[ b ].down();
    }

    /*
     * not thread safe!
     */
    bool notify_event( EventID id )
    {
        assert( events.count( id ) );

        if( events[ id ].is_reached() )
        {
            for( auto & follower : events[ id ].followers )
            {
                events[ follower ].down();
                notify_event( follower );
            }

            events.erase( id );

            return true;
        }
        else
            return false;
    }

public:
    bool empty()
    {
        std::lock_guard< std::mutex > lock( mutex );
        return events.size() == 0;
    }

    /*!
     * Checks whether a event is reached
     */
    bool is_event_reached( EventID event_id )
    {
        std::lock_guard< std::mutex > lock( mutex );

        if( events.count( event_id ) )
            return events[ event_id ].is_reached();
        else
            return true;
    }

    /*!
     * Create a new event without dependencies
     */
    EventID new_event()
    {
        std::lock_guard< std::mutex > lock( mutex );
        return make_event( );
    }

    /*!
     * Creates a new event which precedes the tasks post-event
     */
    EventID add_post_dependency( TaskID task_id )
    {
        std::lock_guard< std::mutex > lock( mutex );
        assert( task_events.count( task_id ) );

        EventID event_id = make_event();
        add_edge( event_id, task_events[ task_id ].post_event );

        return event_id;
    }

    /*!
     * remove the initial dependency on this event
     */
    bool reach_event( EventID event_id )
    {
        std::unique_lock< std::mutex > lock( mutex );

        assert( events.count( event_id ) );
        events[ event_id ].down();

        return notify_event( event_id );
    }

    /*!
     * Checks whether the tasks pre-event is already reached
     */
    bool is_task_ready( TaskID task_id )
    {
        std::lock_guard< std::mutex > lock( mutex );

        assert( task_events.count( task_id ) );

        auto event_id = task_events[ task_id ].pre_event;
        if( events.count( event_id ) )
            return events[ event_id ].is_reached();
        else
            return true;
    }

    /*!
     * Checks whether the tasks post-event is already reached
     */
    bool is_task_finished( TaskID task_id )
    {
        std::lock_guard< std::mutex > lock( mutex );

        if( task_events.count( task_id ) )
            if( events.count( task_events[ task_id ].post_event ) )
                return events[ task_events[ task_id ].post_event ].is_reached();

        return true;
    }

    /*!
     * Insert a new task and add the same dependencies as in the
     * precedence graph
     *
     * Here we assume that the precedence graph is locked
     */
    void add_task( TaskPtr task_ptr )
    {
        auto & task = task_ptr.get();

        std::experimental::optional< TaskID > parent_id;
        if( task.parent )
            parent_id = task.parent->locked_get().task_id;

        std::unique_lock< std::mutex > lock( mutex );
        task_events[ task.task_id ].pre_event = make_event();
        task_events[ task.task_id ].post_event = make_event();

        task.hook_pause(
            [this, task_id=task.task_id]( EventID event_id )
            {
                std::lock_guard< std::mutex > lock( mutex );
                task_events[ task_id ].pre_event = event_id;
            });

        task.hook_before(
            [this, task_id=task.task_id]
            {
                assert( is_task_ready( task_id ) );

                std::lock_guard< std::mutex > lock( mutex );
                if( events.count( task_events[ task_id ].pre_event ) )
                    notify_event( task_events[ task_id ].pre_event );
            });

        task.hook_after(
            [this, task_id=task.task_id]
            {
                std::lock_guard< std::mutex > lock( mutex );

                assert( task_events.count( task_id ) );

                EventID event_id = task_events[ task_id ].post_event;
                events[ event_id ].down();
                notify_event( event_id );

                task_events.erase( task_id );
            });

        for(
            auto it = boost::in_edges( task_ptr.vertex, task_ptr.graph->graph() );
            it.first != it.second;
            ++ it.first
        )
        {
            auto & preceding_task =
               graph_get(
                   boost::source( *(it.first), task_ptr.graph->graph() ),
                   task_ptr.graph->graph()
               ).first;

            if( task_events.count( preceding_task.task_id ) )
                if( events.count( task_events[ preceding_task.task_id ].post_event ) )
                    add_edge(
                        task_events[ preceding_task.task_id ].post_event,
                        task_events[ task.task_id ].pre_event
                    );
        }

        if( parent_id )
        {
            assert( task_events.count( *parent_id ) );
            assert( events.count( task_events[ *parent_id ].post_event ) );
            add_edge(
                task_events[ task.task_id ].post_event,
                task_events[ *parent_id ].post_event
            );
        }

        events[ task_events[ task.task_id ].pre_event ].down();
    }

    /*!
     * remove revoked dependencies 
     * (e.g. after access demotion)
     */
    auto update_task( TaskPtr const & task_ptr )
    {
        auto lock = task_ptr.graph->unique_lock();
        auto vertices = task_ptr.graph->update_vertex( task_ptr.vertex );
        auto task_id = task_ptr.get().task_id;

        {
            std::lock_guard< std::mutex > lock( mutex );
            for( auto v : vertices )
            {
                auto other_task_id = graph_get(v, task_ptr.graph->graph()).first.task_id;
                remove_edge(
                    task_events[ task_id ].post_event,
                    task_events[ other_task_id ].pre_event
                );
            }

            for( auto v : vertices )
            {
                auto other_task_id = graph_get(v, task_ptr.graph->graph()).first.task_id;
                notify_event( task_events[ other_task_id ].pre_event );
            }
        }

        return vertices;
    }

}; // class SchedulingGraph
    
} // namespace redGrapes

