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

#include <redGrapes/task/property/id.hpp>
#include <redGrapes/scheduler/event.hpp>

namespace redGrapes
{
namespace scheduler
{

/*!
 * Each task associates with two events:
 * A Pre-Event and a Post-Event.
   \verbatim
                     +------+
   >>> /  Pre- \ >>> | Task | >>> / Post- \ >>>
       \ Event /     +------+     \ Event /

   \endverbatim
 *
 * Edges between Events determine in which order tasks
 * can be scheduled.
 *
 * Data-dependencies between tasks are assured by
 * edges from post-events to pre-events.
 *
 * With child-tasks, the post-event of the child task
 * precedes the parent tasks post-event.
 */
struct SchedulingGraphProp
{
    std::shared_ptr< Event > pre_event;
    std::shared_ptr< Event > post_event;

    bool is_ready() { return pre_event->is_ready(); }
    bool is_running() { return pre_event->is_reached(); }
    bool is_finished() { return post_event->is_reached(); }

    /*! create a new event which precedes the tasks post-event
     */
    std::shared_ptr< Event > make_event()
    {
        std::shared_ptr<Event> event = std::make_shared< Event >();
        event->add_follower( post_event );
        return event;
    }

    /*!
     * represent ›pausation of the task until event is reached‹
     * in the scheduling graph
     */
    void sg_pause( std::shared_ptr< Event > event )
    {
        pre_event = std::make_shared< Event >();
        SPDLOG_TRACE("sg pause: new_event = {}", (void*) event.get());
        event->add_follower(pre_event);
    }

    /*!
     * Insert a new task and add the same dependencies as in the precedence graph.
     * Note that tasks must be added in order, since only preceding tasks are considered!
     *
     * The precedence graph containing the task is assumed to be locked.
     */
    template < typename Task, typename RedGrapes >
    void sg_init( RedGrapes & rg, TaskVertexPtr task_vertex )
    {
        SPDLOG_TRACE("sg init task {}", task_vertex->template get_task<Task>().task_id);
        pre_event = std::make_shared<Event>( task_vertex );
        post_event = std::make_shared<Event>( task_vertex );

        SPDLOG_TRACE("sginit: pre_event = {}", (void*)pre_event.get());
        SPDLOG_TRACE("sginit: post_event = {}", (void*)post_event.get());

        // add dependencies to tasks which precede the new one
        for(auto weak_in_vertex_ptr : task_vertex->in_edges)
        {
            if( auto preceding_task_vertex = weak_in_vertex_ptr.lock() )
            {
                auto & preceding_task = preceding_task_vertex->template get_task<Task>();

                auto preceding_event
                        = rg.get_scheduler()->task_dependency_type(preceding_task_vertex, task_vertex)
                        ? preceding_task.pre_event
                        : preceding_task.post_event;

                if( preceding_event )
                    if(! preceding_event->is_reached() )
                        preceding_event->add_follower(pre_event);
            }
        }

        // add dependency to parent
        if( auto parent = task_vertex->space.lock()->parent )
        {
            auto & parent_task = parent->lock()->template get_task<Task>();
            post_event->add_follower( parent_task.post_event );
        }
    }

    /*! remove revoked dependencies (e.g. after access demotion)
     *
     * @param revoked_followers set of tasks following this task
     *                          whose dependency on it got removed
     *
     * The precedence graph containing task_vertex is assumed to be locked.
     */
    template < typename Task, typename RedGrapes >
    void sg_revoke_followers( RedGrapes & rg, TaskVertexPtr task_vertex, std::vector<TaskVertexPtr> revoked_followers )
    {
        for( auto follower_vertex : revoked_followers )
        {
            if( ! rg.get_scheduler()->task_dependency_type( task_vertex, follower_vertex ) )
            {
                auto event = follower_vertex->get_task<Task>().pre_event;
                post_event->remove_follower( event );
                rg.notify_event( event );
            }
            // else: the pre-event of task_vertex's task shouldn't exist at this point, so we do nothing            
        }
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


