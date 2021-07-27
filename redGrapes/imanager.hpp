/* Copyright 2019-2021 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <memory>
#include <optional>

#include <redGrapes/task/task_space.hpp>

namespace redGrapes
{
    template<typename Task>
    struct IManager;
}

#include <redGrapes/graph/scheduling_graph.hpp>
#include <redGrapes/scheduler/scheduler.hpp>

namespace redGrapes
{
    template<typename Task>
    struct IManager
    {
        using TaskVertexPtr = std::shared_ptr<PrecedenceGraphVertex<Task>>;
        
        virtual ~IManager() {}
        
        virtual std::shared_ptr<SchedulingGraph<Task>> get_scheduling_graph() {}
        virtual std::shared_ptr<scheduler::IScheduler<Task>> get_scheduler() {}
        virtual std::shared_ptr<TaskSpace<Task>> get_main_space() {}
        virtual std::shared_ptr<TaskSpace<Task>> current_task_space() {}
        virtual std::optional<TaskVertexPtr>& current_task() {}

        virtual void activate_task(TaskVertexPtr) {}
        virtual bool activate_next() {}

        virtual void yield(EventID event_id) {}
        virtual void remove_task(TaskVertexPtr) {}

        virtual std::optional<EventID> create_event() {}
        virtual void reach_event(EventID event_id) {}
    };
}

