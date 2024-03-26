/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/scheduler/event.hpp"
#include "redGrapes/task/task_space.hpp"
#include "redGrapes/util/trace.hpp"

#include <spdlog/spdlog.h>

#include <optional>

namespace redGrapes
{
    template<typename TTask>
    struct TaskCtx
    {
        //! pause the currently running task at least until event is reached
        // else is supposed to be called when .get() is called on emplace task, which calls the future .get(), so there
        // is no current task at that time, unless this is in a child task space. we can assert(event.task != 0);
        static void yield(scheduler::EventPtr<TTask> event)
        {
            if(current_task)
            {
                while(!event->is_reached())
                    current_task->yield(event);
            }
            else
            {
                event->waker_id = event.task->scheduler.getNextWorkerID();
                while(!event->is_reached())
                    TaskFreeCtx::idle();
            }
        }

        /*! Create an event on which the termination of the current task depends.
         *  A task must currently be running.
         *
         * @return Handle to flag the event with `reach_event` later.
         *         nullopt if there is no task running currently
         */
        static std::optional<scheduler::EventPtr<TTask>> create_event()
        {
            if(current_task)
                return current_task->make_event();
            else
                return std::nullopt;
        }

        static std::shared_ptr<TaskSpace<TTask>> current_task_space()
        {
            if(current_task)
            {
                if(!current_task->children)
                {
                    auto task_space = std::make_shared<TaskSpace<TTask>>(current_task);
                    SPDLOG_TRACE("create child space = {}", (void*) task_space.get());
                    current_task->children = task_space;

                    std::unique_lock<std::shared_mutex> wr_lock(current_task->space->active_child_spaces_mutex);
                    current_task->space->active_child_spaces.push_back(task_space);
                }

                return current_task->children;
            }
            else
                return root_space;
        }

        static unsigned scope_depth()
        {
            if(auto ts = current_task_space())
                return ts->depth;
            else
                return 0;
        }

        static inline thread_local TTask* current_task;
        static inline std::shared_ptr<TaskSpace<TTask>> root_space;


#if REDGRAPES_ENABLE_TRACE
        static std::shared_ptr<perfetto::TracingSession> tracing_session;
#endif
    };

} // namespace redGrapes
