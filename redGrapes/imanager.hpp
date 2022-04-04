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
    struct IManager;
}

#include <redGrapes/scheduler/scheduling_graph.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/task_space.hpp>

namespace redGrapes
{
    struct IManager
    {
        virtual ~IManager() {}

        virtual std::shared_ptr<scheduler::IScheduler> get_scheduler() {}
        virtual std::shared_ptr<TaskSpace> get_main_space() {}
        virtual std::shared_ptr<TaskSpace> current_task_space() {}
        virtual std::optional<TaskVertexPtr>& current_task() {}

        virtual void update_active_task_spaces() {}
        
        virtual void activate_task(TaskVertexPtr) {}
        virtual bool activate_next() {}

        virtual void notify_event(scheduler::EventPtr event) {}
        virtual std::optional<scheduler::EventPtr> create_event() {}
        virtual void yield(scheduler::EventPtr event) {}
        virtual void remove_task(TaskVertexPtr) {}

    };
}

