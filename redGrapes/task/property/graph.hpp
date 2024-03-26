/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/scheduler/event.hpp"

#include <spdlog/spdlog.h>

#include <memory>

namespace redGrapes
{

    // struct Task;
    template<typename TTask>
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
    template<typename TTask>
    struct GraphProperty
    {
        TTask& operator*()
        {
            return *task;
        }

        TTask* operator->()
        {
            return task;
        }

        TTask* task;

        //! number of parents
        uint8_t scope_depth;

        //! task space that contains this task, must not be null
        std::shared_ptr<TaskSpace<TTask>> space;

        //! task space for children, may be null
        std::shared_ptr<TaskSpace<TTask>> children;

        /*
        // in edges dont need a mutex because they are initialized
        // once by `init_dependencies()` and only read afterwards.
        // expired pointers (null) must be ignored
        std::vector<Task*> in_edges;
        */

        scheduler::Event<TTask> pre_event;
        scheduler::Event<TTask> post_event;
        scheduler::Event<TTask> result_set_event;
        scheduler::Event<TTask> result_get_event;

        inline scheduler::EventPtr<TTask> get_pre_event()
        {
            return scheduler::EventPtr<TTask>{scheduler::T_EVT_PRE, this->task};
        }

        inline scheduler::EventPtr<TTask> get_post_event()
        {
            return scheduler::EventPtr<TTask>{scheduler::T_EVT_POST, this->task};
        }

        inline scheduler::EventPtr<TTask> get_result_set_event()
        {
            return scheduler::EventPtr<TTask>{scheduler::T_EVT_RES_SET, this->task};
        }

        inline scheduler::EventPtr<TTask> get_result_get_event()
        {
            return scheduler::EventPtr<TTask>{scheduler::T_EVT_RES_GET, this->task};
        }

        inline bool is_ready()
        {
            return pre_event.is_ready();
        }

        inline bool is_running()
        {
            return pre_event.is_reached();
        }

        inline bool is_finished()
        {
            return post_event.is_reached();
        }

        inline bool is_dead()
        {
            return post_event.is_reached() && result_get_event.is_reached();
        }

        /*! create a new event which precedes the tasks post-event
         */
        scheduler::EventPtr<TTask> make_event();

        /*!
         * represent ›pausation of the task until event is reached‹
         * in the scheduling graph
         */
        inline void sg_pause(scheduler::EventPtr<TTask> event)
        {
            pre_event.state = 1;
            event->add_follower(get_pre_event());
        }

        /*!
         * Insert a new task and add the same dependencies as in the precedence graph.
         * Note that tasks must be added in order, since only preceding tasks are considered!
         *
         * The precedence graph containing the task is assumed to be locked.
         */
        void init_graph();

        /*!
         * Abstractly adds a dependeny from preceding task to this,
         * by setting up an edge from the post-event of the
         * preceding task to the pre-event of this task.
         * Additionally, an edge to the post-event of the parent is added.
         */
        void add_dependency(TTask& preceding_task);

        /*!
         * checks all incoming edges if they are still required and
         * removes them if possible.
         */
        void update_graph();

        /*!
         * removes this task from all resource-user-lists, so from now on
         * no new dependencies to this task will be created.
         */
        void delete_from_resources();

        template<typename PropertiesBuilder>
        struct Builder
        {
            PropertiesBuilder& builder;

            Builder(PropertiesBuilder& b) : builder(b)
            {
            }
        };

        struct Patch
        {
            template<typename PatchBuilder>
            struct Builder
            {
                Builder(PatchBuilder&)
                {
                }
            };
        };

        void apply_patch(Patch const&){};
    };

} // namespace redGrapes

template<typename TTask>
struct fmt::formatter<redGrapes::GraphProperty<TTask>>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::GraphProperty<TTask> const& sg_prop, FormatContext& ctx)
    {
        return ctx.out();
    }
};
