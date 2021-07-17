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
#include <redGrapes/graph/scheduling_graph.hpp>

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

    using TaskPtr = std::shared_ptr<PrecedenceGraphVertex<Task>>;

    /*! whats the task dependency type for the edge a -> b (task a precedes task b)
     * @return true if task b depends on the pre event of task a, false if task b depends on the post event of task b.
     */
    virtual bool task_dependency_type( TaskPtr a, TaskPtr b )
    {
        return false;
    }

    /*! Tell the scheduler to consider dispatching a task.
     */
    virtual void activate_task( TaskPtr task_ptr ) = 0;

    /*! Notify the scheduler that the scheduling graph has changed.
     * The scheduler should now reconsider activated tasks which were not ready before
     */
    virtual void notify() {}
};

} // namespace scheduler

} // namespace redGrapes

