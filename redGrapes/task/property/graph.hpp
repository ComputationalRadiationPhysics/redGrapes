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
#include <optional>
#include <spdlog/spdlog.h>

#include <memory>

#include <redGrapes/task/itask.hpp>
#include <redGrapes/task/property/inherit.hpp>
#include <redGrapes/scheduler/event.hpp>

namespace redGrapes
{

struct Task;
struct TaskSpace;

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
struct GraphProperty
{
    Task & operator*()
    {
        return *task;
    }
    Task * operator->()
    {
        return task;
    }

    Task * task;

    // ticket for initialization of dependency graph
    unsigned ticket;
    
    //! number of parents
    unsigned int scope_depth;

    //! task space that contains this task, must not be null
    std::shared_ptr< TaskSpace > space;

    //! task space for children, may be null
    std::shared_ptr< TaskSpace > children;

    // in edges dont need a mutex because they are initialized
    // once by `init_dependencies()` and only read afterwards.
    // expired pointers must be ignored
    std::vector<Task*> in_edges;

    scheduler::Event pre_event;
    scheduler::Event post_event;
    scheduler::Event result_set_event;
    scheduler::Event result_get_event;

    scheduler::EventPtr get_pre_event();
    scheduler::EventPtr get_post_event();
    scheduler::EventPtr get_result_set_event();
    scheduler::EventPtr get_result_get_event();

    bool is_ready();
    bool is_running();
    bool is_finished();
    bool is_dead();

    /*! create a new event which precedes the tasks post-event
     */
    scheduler::EventPtr make_event();

    /*!
     * represent ›pausation of the task until event is reached‹
     * in the scheduling graph
     */
    void sg_pause( scheduler::EventPtr event );

    /*!
     * Insert a new task and add the same dependencies as in the precedence graph.
     * Note that tasks must be added in order, since only preceding tasks are considered!
     *
     * The precedence graph containing the task is assumed to be locked.
     */
    void init_graph();
    void add_dependency( Task & preceding_task );
    void update_graph();
    void delete_from_resources();

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

} // namespace redGrapes

template <>
struct fmt::formatter< redGrapes::GraphProperty >
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::GraphProperty const & sg_prop,
        FormatContext & ctx
    )
    {
        return ctx.out();
    }
};

