/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <functional>
#include <memory>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/scheduler/scheduling_graph.hpp>

namespace redGrapes
{
namespace scheduler
{

/*! Scheduler Interface
 */
template < typename Task >
struct IScheduler
{
    virtual ~IScheduler() {}

    /*! whats the task dependency type for the edge a -> b (task a precedes task b)
     * @return true if task b depends on the pre event of task a, false if task b depends on the post event of task b.
     */
    virtual bool task_dependency_type( TaskVertexPtr a, TaskVertexPtr b )
    {
        return false;
    }

    virtual void activate_task( TaskVertexPtr task_vertex ) {}

    //! wakeup to call activate_next()
    virtual void notify() {}
};

} // namespace scheduler

} // namespace redGrapes

