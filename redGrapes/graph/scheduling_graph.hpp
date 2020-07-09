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

namespace redGrapes
{

/*!
 * Manages a flat, non-recursive graph of events.
 * An event is the abstraction of the programs execution state.
 * During runtime, each thread encounters a sequence of events.
 * The goal is to synchronize these events in the manner
 * "Event A must occur before Event B".
 *
 * Multiple events need to be related, so that they
 * form a partial order (i.e. an antisymmetric quasiorder).
 * This order is an homomorph image from the timeline of
 * execution states.
 *
 *
 * Each task is represented by at least two events:
 * A Pre-Event and a Post-Event.
   \verbatim
                     +------+
   >>> /  Pre- \ >>> | Task | >>> / Post- \ >>>
       \ Event /     +------+     \ Event /

   \endverbatim
 *
 * Data-dependencies between tasks are assured by
 * edges from post-events to pre-events.
 *
 * Child-tasks are inserted, so that the child tasks post-event
 * precedes the parent tasks post-event.
 */
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
        /*! number of incoming edges
         * state == 0: event is reached and can be removed
         */
        unsigned int state;

        //! the set of subsequent events
        std::vector< EventID > followers;

        Event()
            // every event needs at least one down() before it will be removed
            : state( 1 )
        {}

        bool is_reached() { return state == 0; }
        void up() { state += 1; }
        void down() { state -= 1; }
    };

    struct TaskEvents
    {
        TaskID pre_event;
        TaskID post_event;
    };

    std::mutex mutex;
    EventID event_id_counter;
    std::unordered_map< EventID, Event > events;
    std::unordered_map< TaskID , TaskEvents > task_events;

    /*!
     * Create a new event (with no dependencies)
     *
     * Not thread safe!
     */
    EventID make_event()
    {
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
            std::find( std::begin(fs), std::end(fs), b )
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

    std::function< bool( TaskPtr, TaskPtr ) > task_dependency_type;

public:
    /*
     * It can be configured, how task dependencies are represented in the scheduling graph
     * using the dependency_event_type function. This is useful for representing asynchronous
     * tasks whose dependencies are managed externally, e.g. asynchronous CUDA operations.
     * In the case of CUDA, the pre-event of a task can be used to represent the submission
     * of the asynchronous call.
     *
     * @param task_dependency_type function to determine whether to take the preceding tasks
     *                             pre- or post-event as dependency. By returning true, an edge
     *                             to the preceding tasks pre-event is added. By default, false
     *                             is returned, meaning an edge to the post-event.
     */
    SchedulingGraph(
        std::function< bool ( TaskPtr, TaskPtr ) >
        task_dependency_type
            = [] ( TaskPtr, TaskPtr )
              {
                  return false;
              }
    ) :
        task_dependency_type( task_dependency_type ),
        event_id_counter( 0 )
    {}

    //! are all events reached?
    bool empty()
    {
        std::lock_guard< std::mutex > lock( mutex );
        return events.size() == 0;
    }

    //! checks whether an event is reached
    bool is_event_reached( EventID event_id )
    {
        std::lock_guard< std::mutex > lock( mutex );

        if( events.count( event_id ) )
            return events[ event_id ].is_reached();
        else
            return true;
    }

    //! create a new event without dependencies
    EventID new_event()
    {
        std::lock_guard< std::mutex > lock( mutex );
        return make_event( );
    }

    //! creates a new event which precedes the tasks post-event
    EventID add_post_dependency( TaskID task_id )
    {
        std::lock_guard< std::mutex > lock( mutex );
        assert( task_events.count( task_id ) );

        EventID event_id = make_event();
        add_edge( event_id, task_events[ task_id ].post_event );

        return event_id;
    }

    //! remove the initial dependency on this event
    bool reach_event( EventID event_id )
    {
        std::unique_lock< std::mutex > lock( mutex );

        assert( events.count( event_id ) );
        events[ event_id ].down();

        return notify_event( event_id );
    }

    //! checks whether the tasks pre-event is already reached
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

    //! checks whether the tasks post-event is already reached
    bool is_task_finished( TaskID task_id )
    {
        std::lock_guard< std::mutex > lock( mutex );

        if( task_events.count( task_id ) )
            if( events.count( task_events[ task_id ].post_event ) )
                return events[ task_events[ task_id ].post_event ].is_reached();

        return true;
    }

    //! notify the tasks pre-event
    void task_start( TaskID task_id )
    {
        assert( is_task_ready( task_id ) );

        std::lock_guard< std::mutex > lock( mutex );
        if( events.count( task_events[ task_id ].pre_event ) )
            notify_event( task_events[ task_id ].pre_event );
    }

    //! notify the tasks post-event
    void task_end( TaskID task_id )
    {
        std::lock_guard< std::mutex > lock( mutex );

        assert( task_events.count( task_id ) );

        EventID event_id = task_events[ task_id ].post_event;
        events[ event_id ].down();
        notify_event( event_id );
    }

    //! pause the task until event_id is reached
    void task_pause( TaskID task_id, EventID event_id )
    {
        std::lock_guard< std::mutex > lock( mutex );
        task_events[ task_id ].pre_event = event_id;
    }

    void remove_task( TaskID task_id )
    {
        assert( is_task_finished( task_id ) );

        std::lock_guard< std::mutex > lock( mutex );
        task_events.erase( task_id );
    }

    /*!
     * Insert a new task and add the same dependencies as in the precedence graph.
     * Note that tasks must be added in order, since only preceding tasks are considered!
     *
     * The precedence graph containing the task is assumed to be locked.
     */
    void add_task( TaskPtr task_ptr )
    {
        auto & task = task_ptr.get();

        std::experimental::optional< TaskID > parent_id;
        if( task.parent )
            parent_id = task.parent->locked_get().task_id;

        // create new events for the task
        std::unique_lock< std::mutex > lock( mutex );
        task_events[ task.task_id ].pre_event = make_event();
        task_events[ task.task_id ].post_event = make_event();

        // add dependencies to tasks which precede the new one
        for(
            auto it = boost::in_edges( task_ptr.vertex, task_ptr.graph->graph() );
            it.first != it.second;
            ++ it.first
        )
        {
            TaskPtr preceding_task_ptr
            {
                task_ptr.graph,
                boost::source( *(it.first), task_ptr.graph->graph() )
            };

            auto & preceding_task_id = preceding_task_ptr.get().task_id;

            if( task_events.count( preceding_task_id ) )
            {
                EventID preceding_event_id =
                    task_dependency_type( preceding_task_ptr, task_ptr ) ?
                        task_events[ preceding_task_id ].pre_event
                    :
                        task_events[ preceding_task_id ].post_event;

                if( events.count( preceding_event_id ) )
                    add_edge(
                        preceding_event_id,
                        task_events[ task.task_id ].pre_event
                    );
            }
        }

        // add dependency to parent
        if( parent_id )
        {
            assert( task_events.count( *parent_id ) );
            assert( events.count( task_events[ *parent_id ].post_event ) );
            add_edge(
                task_events[ task.task_id ].post_event,
                task_events[ *parent_id ].post_event
            );
        }
        // else: task has no parent

        events[ task_events[ task.task_id ].pre_event ].down();
    }

    //! remove revoked dependencies (e.g. after access demotion)
    void update_task( TaskPtr const & task_ptr, std::vector< TaskPtr > const & followers )
    {
        auto task_id = task_ptr.get().task_id;

        {
            std::lock_guard< std::mutex > lock( mutex );

            for( auto other_task_ptr : followers )
            {
                auto other_task_id = other_task_ptr.get().task_id;

                if( ! task_dependency_type( task_ptr, other_task_ptr ) )
                {
                    remove_edge(
                        task_events[ task_id ].post_event,
                        task_events[ other_task_id ].pre_event
                    );

                    notify_event( task_events[ other_task_id ].pre_event );                    
                }
                // else: the pre-event of task_ptr's task shouldn't exist at this point, so we do nothing
            }
        }
    }

}; // class SchedulingGraph
    
} // namespace redGrapes

