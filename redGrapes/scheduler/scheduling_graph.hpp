/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <cassert>
#include <spdlog/spdlog.h>

#include <redGrapes/property/id.hpp>
//#include <redGrapes/imanager.hpp>

namespace std
{
    using shared_mutex = shared_timed_mutex;
} // namespace std

namespace redGrapes
{
namespace scheduler
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
struct Event
{
    /*! number of incoming edges
     * state == 0: event is reached and can be removed
     */
    std::atomic_int state;

    //! the set of subsequent events
    std::vector< std::shared_ptr<Event> > followers;
    std::shared_mutex followers_mutex;
    std::recursive_mutex mutex;

    std::function< void() > ready_hook;
    std::function< void() > reach_hook;

    Event()
        : state(1)
    {}

    Event( std::function< void() > ready_hook )
        : state(1)
        , ready_hook(ready_hook)
    {}

    bool is_reached() { return state == 0; }
    bool is_ready() { return state == 1; }
    void up() { state++; }
    void dn() { state--; }

    void add_follower( std::shared_ptr<Event> follower )
    {
        SPDLOG_TRACE("event {} add follower {}", (void*)this, (void*)follower.get());
        std::unique_lock< std::shared_mutex > lock( followers_mutex );
        follower->up();
        followers.push_back(follower);
    }

    void remove_follower( std::shared_ptr<Event> follower )
    {
        std::unique_lock< std::shared_mutex > lock( followers_mutex );
        followers.erase(
            std::find( std::begin(followers), std::end(followers), follower )
        );
        follower->dn();
    }

    void reach()
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );

        dn();
        notify();
    }

    template < typename F >
    void set_ready_hook( F && f )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        if( !is_ready() )
        {
            if(! ready_hook)
                ready_hook = f;
            else
                throw std::runtime_error("set_ready_hook: already set!");
        }
        else
            f();
        /*
        else
            throw std::runtime_error("set_ready_hook: event already reached!");
        */
    }
    
    template < typename F >
    void set_reach_hook( F && f )
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        if( !is_reached() )
        {
            if(! reach_hook)
                reach_hook = f;
            else
                throw std::runtime_error("set_reach_hook: already set!");
        }
        else
            throw std::runtime_error("set_reach_hook: event already reached!");
    }

    bool notify()
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );
        if( is_ready() )
        {
            if( ready_hook )
                ready_hook();
        }

        if( is_reached() )
        {
            if( reach_hook )
                reach_hook();
        }

        if( is_reached() )
        {
            // notify followers
            std::shared_lock< std::shared_mutex > lock( followers_mutex );
            for( auto & follower : followers )
                follower->reach();

            return true;
        }
        else
            return false;
    }
};

struct SchedulingGraphProp
{
    std::function< void() > ready_hook;

    std::shared_ptr<Event> pre_event;
    std::shared_ptr<Event> post_event;

    bool is_ready()
    {
        return pre_event->is_ready();
    }

    bool is_running()
    {
        return pre_event->is_reached();
    }

    bool is_finished()
    {
        return post_event->is_reached();
    }

    //! creates a new event which precedes the tasks post-event
    std::shared_ptr<Event> make_event()
    {
        std::shared_ptr<Event> event = std::make_shared< Event >();
        event->add_follower( post_event );
        return event;
    }

    //! pause the task until event is reached
    void pause( std::shared_ptr<Event> event )
    {
        pre_event = std::make_shared< Event >( ready_hook );

        if( !event->is_reached() )
            event->add_follower(pre_event);
        else
            pre_event->notify();
    }

    /*!
     * Insert a new task and add the same dependencies as in the precedence graph.
     * Note that tasks must be added in order, since only preceding tasks are considered!
     *
     * The precedence graph containing the task is assumed to be locked.
     */
    template < typename TaskVertexPtr, typename PostHook >
    void init_scheduling_graph(
        TaskVertexPtr task_ptr,
        PostHook post_hook
    )
    {
        pre_event = std::make_shared<Event>();
        post_event = std::make_shared<Event>();

        post_event->set_reach_hook( post_hook );

        // add dependencies to tasks which precede the new one
        for(auto weak_in_vertex_ptr : task_ptr->in_edges)
        {
            if( auto preceding_task_ptr = weak_in_vertex_ptr.lock() )
            {
                auto & preceding_task = preceding_task_ptr->task;
                /*
                auto preceding_event
                        = mgr.get_scheduler()->task_dependency_type(preceding_task_ptr, task_ptr)
                        ? preceding_task->pre_event
                        : preceding_task->post_event;
                */
                auto preceding_event = preceding_task->post_event;
                if( preceding_event )
                    if(! preceding_event->is_reached() )
                        preceding_event->add_follower(pre_event);
            }
        }

        // add dependency to parent
        if( auto parent = task_ptr->space.lock()->parent )
        {
            parent->lock()->task->post_event->add_follower(post_event);
        }

        pre_event->set_ready_hook(ready_hook);
    }

    /*! remove revoked dependencies (e.g. after access demotion)
     *
     * @param task_ptr the demoted task
     * @param followers set of tasks following task_ptr
     *                  whose dependency on it got removed
     *
     * The precedence graph containing task_ptr is assumed to be locked.
     */
    template < typename Task >
    void update_scheduling_graph(
        //IManager<Task> & mgr,
        typename Task::VertexPtr task_ptr
    )
    {
        // TODO
        /*
        auto task_id = task_ptr.get().task_id;

        {
            std::lock_guard< std::recursive_mutex > lock( followers_mutex );

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
        */
    }

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

} // namespace scheduler

} // namespace redGrapes


template <>
struct fmt::formatter< redGrapes::scheduler::SchedulingGraphProp >
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::scheduler::SchedulingGraphProp const & sg_prop,
        FormatContext & ctx
    )
    {
        return ctx.out();
    }
};


