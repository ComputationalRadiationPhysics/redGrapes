/* Copyright 2019-2021 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <memory>
#include <optional>

namespace redGrapes
{
    template<typename Task>
    struct IManager;
}

#include <redGrapes/scheduler/scheduling_graph.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/task/task_space.hpp>

namespace redGrapes
{
    template<typename Task>
    struct IManager
    {
        using TaskVertexPtr = std::shared_ptr<PrecedenceGraphVertex<Task>>;
        
        virtual ~IManager() {}

        virtual std::shared_ptr<scheduler::IScheduler<Task>> get_scheduler() {}
        virtual std::shared_ptr<TaskSpace<Task>> get_main_space() {}
        virtual std::shared_ptr<TaskSpace<Task>> current_task_space() {}
        virtual std::optional<TaskVertexPtr>& current_task() {}

        virtual void update_active_task_spaces() {}
        
        virtual void activate_task(TaskVertexPtr) {}
        virtual bool activate_next() {}

        virtual std::optional<std::shared_ptr<scheduler::Event>> create_event() {}
        virtual void yield(std::shared_ptr<scheduler::Event> event_id) {}
        virtual void remove_task(TaskVertexPtr) {}

    };
}

