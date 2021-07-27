/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <cassert>
#include <redGrapes/property/id.hpp>
#include <spdlog/spdlog.h>

namespace redGrapes
{
    using EventID = unsigned int;

    template<typename Task>
    struct SchedulingGraph;
} // namespace redGrapes

#include <redGrapes/imanager.hpp>

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
 * This order is an homomorphic image from the timeline of
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
    typename Task
>
class SchedulingGraph
{
private:
    struct Event
    {
        /*! number of incoming edges
         * state == 0: event is reached and can be removed
         */
        unsigned int state;

        //! the set of subsequent events
        std::vector< EventID > followers;

        //! task which will be activated when event is reached
        std::optional< typename Task::VertexPtr > task_ptr;
        
        Event()
            : state(1)
        {}

        Event( std::optional< typename Task::VertexPtr > && task_ptr )
            // every event needs at least one down() before it will be removed
            : state( 1 )
            , task_ptr( task_ptr )
        {}

        bool is_reached() { return state == 0; }
        bool is_ready() { return state == 1; }
        void up() { state += 1; }
        void down() { state -= 1; }
    };

    struct TaskEvents
    {
        EventID pre_event;
        EventID post_event;
        typename Task::VertexPtr task_ptr;
    };

    std::recursive_mutex mutex;
    EventID event_id_counter;
    std::unordered_map< EventID, Event > events;
    std::unordered_map< TaskID , TaskEvents > task_events;

    IManager< Task >& mgr;

    /*!
     * Create a new event (with no dependencies)
     *
     * Not thread safe!
     */
    EventID make_event( std::optional< typename Task::VertexPtr > task_ptr = std::nullopt )
    {
        EventID event_id = event_id_counter ++;

        events.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(event_id),
            std::forward_as_tuple(std::move(task_ptr))
        );

        spdlog::trace("SchedulingGraph::make_event() = {}", event_id);

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

        spdlog::trace("SchedulingGraph::add_edge(event {} -> event {})", a, b);

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
        spdlog::trace("SchedulingGraph::notify_event({})", id);
        assert( events.count( id ) );

        if( events[ id ].is_ready() )
            // activate associated task
            if( auto task = events[ id ].task_ptr )
                mgr.activate_task( *task );

        if( events[ id ].is_reached() )
        {
            // notify followers
            for( auto & follower : events[ id ].followers )
                unsafe_reach_event( follower );

            spdlog::trace("SchedulingGraph: remove event {}", id);
            events.erase( id );

            return true;
        }
        else
            return false;
    }

    /*
     * not thread safe!
     */
    bool unsafe_reach_event( EventID event_id )
    {
        spdlog::trace("SchedulingGraph::unsafe_reach_event({})", event_id);

        if( events.count( event_id ) )
        {
            events[ event_id ].down();
            return notify_event( event_id );
        }
        else
            return false;
    }

    using TaskPtr = typename Task::VertexPtr;

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
    SchedulingGraph( IManager< Task > & mgr )
        : mgr( mgr )
        , event_id_counter( 0 )
    {}

    //! are all events reached?
    bool empty()
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );        
        return events.size() == 0;
    }

    bool unsafe_is_event_reached( EventID event_id )
    {
        if( events.count( event_id ) )
            return events[ event_id ].is_reached();
        else
            return true;        
    }
    
    //! checks whether an event is reached
    bool is_event_reached( EventID event_id )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        return unsafe_is_event_reached( event_id );
    }

    //! create a new event without dependencies
    EventID new_event()
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        return make_event( );
    }

    //! creates a new event which precedes the tasks post-event
    EventID add_post_dependency( TaskID task_id )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        assert( task_events.count( task_id ) );

        EventID event_id = make_event();
        add_edge( event_id, task_events[ task_id ].post_event );

        return event_id;
    }

    //! remove the initial dependency on this event
    bool reach_event( EventID event_id )
    {
        std::unique_lock< std::recursive_mutex > lock( mutex );
        return unsafe_reach_event( event_id );
    }

    //! checks whether the tasks pre-event is already ready
    bool is_task_ready( TaskID task_id )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );

        if( task_events.count( task_id ) )
        {
            auto event_id = task_events[ task_id ].pre_event;
            if( events.count( event_id ) )
                return events[ event_id ].is_ready();
        }

        return false;
    }

    //! checks whether the tasks pre-event exists
    bool is_task_running( TaskID task_id )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        if( task_events.count( task_id ) )
            return is_event_reached( task_events[ task_id ].pre_event );
        else
            return false;
    }

    //! checks whether the tasks post-event is already reached
    bool is_task_finished( TaskID task_id )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );

        if( task_events.count( task_id ) )
            if( events.count( task_events[ task_id ].post_event ) )
                return events[ task_events[ task_id ].post_event ].is_reached();

        return true;
    }
 
    //! notify the tasks pre-event
    void task_start( TaskID task_id )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        spdlog::trace("SchedulingGraph::task_start({})", task_id);

        assert( task_events.count(task_id) );
        assert( events.count( task_events[ task_id ].pre_event ) );
        assert( events[ task_events[ task_id ].pre_event ].is_ready() );
        bool r = unsafe_reach_event( task_events[ task_id ].pre_event );
        assert( r );
    }

    //! notify the tasks post-event
    void task_end( TaskID task_id )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        spdlog::trace("SchedulingGraph::task_end({})", task_id);

        assert( task_events.count( task_id ) );
        auto r = unsafe_reach_event( task_events[ task_id ].post_event );
        spdlog::trace("SchedulingGraph: unsafe_reach_event() = {}", r);
    }

    //! pause the task until event_id is reached
    void task_pause( TaskID task_id, EventID event_id )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        spdlog::trace("SchedulingGraph::task_pause({})", task_id);

        task_events[ task_id ].pre_event = make_event( task_events[task_id].task_ptr );

        spdlog::trace("SchedulingGraph::task_pause set pre_event={}", task_events[ task_id ].pre_event);

        if(
           events.count(event_id) &&
           !events[event_id].is_reached()
        )
            add_edge(event_id, task_events[task_id].pre_event);
        else
            // event was reached before task_pause()
            notify_event(task_events[task_id].pre_event);

    }

    void remove_task( TaskID task_id )
    {
        assert( is_task_finished( task_id ) );

        std::lock_guard< std::recursive_mutex > lock( mutex );
        spdlog::trace("SchedulingGraph::remove_task({})", task_id);

        assert( !events.count( task_events[task_id].pre_event ) );
        assert( !events.count( task_events[task_id].post_event ) );
        task_events.erase( task_id );
    }

    bool exists_task( TaskID task_id )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        return task_events.count( task_id );
    }
    
    /*!
     * Insert a new task and add the same dependencies as in the precedence graph.
     * Note that tasks must be added in order, since only preceding tasks are considered!
     *
     * The precedence graph containing the task is assumed to be locked.
     */
    void add_task( TaskPtr task_ptr )
    {
        auto & task = *task_ptr->task;

        std::experimental::optional< TaskID > parent_id;
        if( auto parent = task_ptr->space.lock()->parent )
            parent_id = parent->lock()->task->task_id;

        // create new events for the task
        std::unique_lock< std::recursive_mutex > lock( mutex );

        if( exists_task( task.task_id ) )
             return;

        task_events[ task.task_id ].task_ptr = task_ptr;
        task_events[ task.task_id ].pre_event = make_event( task_ptr );
        task_events[ task.task_id ].post_event = make_event( std::nullopt );

        spdlog::trace(
            "SchedulingGraph::add_task({}) -> pre={}, post={}",
            task.task_id,
            task_events[task.task_id].pre_event,
            task_events[task.task_id].post_event
        );

        // add dependencies to tasks which precede the new one
        for(auto weak_in_vertex_ptr : task_ptr->in_edges)
        {
            if( TaskPtr preceding_task_ptr = weak_in_vertex_ptr.lock() )
            {
                auto preceding_task_id = preceding_task_ptr->task->task_id;
                spdlog::trace("SchedulingGraph: preceding task {}", preceding_task_id);

                if(task_events.count(preceding_task_id))
                {
                    spdlog::trace(
                        "SchedulingGraph: task {} -> task {}: dependency type {}",
                        preceding_task_id,
                        task.task_id,
                        mgr.get_scheduler()->task_dependency_type(preceding_task_ptr, task_ptr));

                    EventID preceding_event_id
                        = mgr.get_scheduler()->task_dependency_type(preceding_task_ptr, task_ptr)
                        ? task_events[preceding_task_id].pre_event
                        : task_events[preceding_task_id].post_event;

                    if(! unsafe_is_event_reached(preceding_event_id))
                        add_edge(preceding_event_id, task_events[task.task_id].pre_event);
                }
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
    }

    /*! remove revoked dependencies (e.g. after access demotion)
     *
     * @param task_ptr the demoted task
     * @param followers set of tasks following task_ptr
     *                  whose dependency on it got removed
     *
     * The precedence graph containing task_ptr is assumed to be locked.
     */
    void update_task(
        TaskPtr const & task_ptr,
        std::vector< TaskPtr > const & followers
    )
    {
        auto task_id = task_ptr.get().task_id;

        {
            std::lock_guard< std::recursive_mutex > lock( mutex );

            for( auto other_task_ptr : followers )
            {
                auto other_task_id = other_task_ptr.get().task_id;

                if( ! mgr.get_scheduler()->task_dependency_type( task_ptr, other_task_ptr ) )
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

