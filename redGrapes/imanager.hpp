/* Copyright 2019-2021 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <memory>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/graph/scheduling_graph.hpp>

namespace redGrapes
{
    template<typename Task>
    struct IManager
    {
        using TaskVertexPtr = std::shared_ptr<PrecedenceGraphVertex<Task>>;
        
        virtual ~IManager() {}
        
        virtual std::shared_ptr< SchedulingGraph<Task> > get_scheduling_graph()
        {
            return 0;
        }

        virtual std::shared_ptr<TaskSpace<Task>> current_task_space() {}
        virtual void activate_task(TaskVertexPtr) {}
        virtual bool run_task(TaskVertexPtr) {}
        virtual void remove_task(TaskVertexPtr) {}

        virtual void notify() {}
    };
}
