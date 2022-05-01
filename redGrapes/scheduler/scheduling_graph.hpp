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

#include <redGrapes/task/itask.hpp>
#include <redGrapes/task/task_space.hpp>
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
    Event pre_event;
    Event post_event;
    Event result_event;

    bool is_ready();
    bool is_running();
    bool is_finished();

    SchedulingGraphProp();
    SchedulingGraphProp(SchedulingGraphProp const &);
    
    /*! create a new event which precedes the tasks post-event
     */
    EventPtr make_event();

    /*!
     * represent ›pausation of the task until event is reached‹
     * in the scheduling graph
     */
    void sg_pause( EventPtr event );

    /*!
     * Insert a new task and add the same dependencies as in the precedence graph.
     * Note that tasks must be added in order, since only preceding tasks are considered!
     *
     * The precedence graph containing the task is assumed to be locked.
     */
    template < typename Task, typename RedGrapes >
    void sg_init( RedGrapes & rg, TaskVertexPtr task_vertex );

    /*! remove revoked dependencies (e.g. after access demotion)
     *
     * @param revoked_followers set of tasks following this task
     *                          whose dependency on it got removed
     *
     * The precedence graph containing task_vertex is assumed to be locked.
     */
    template < typename Task, typename RedGrapes >
    void sg_revoke_followers( RedGrapes & rg, TaskVertexPtr task_vertex, std::vector<TaskVertexPtr> revoked_followers );


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

#include <redGrapes/scheduler/event.cpp>
#include <redGrapes/scheduler/event_ptr.cpp>
#include <redGrapes/scheduler/scheduling_graph.cpp>

