/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <functional>
#include <memory>
#include <redGrapes/graph/scheduling_graph.hpp>

namespace redGrapes
{
namespace scheduler
{

/*! Scheduler Interface
 */
template <
    typename TaskID,
    typename TaskPtr
>
struct IScheduler
{
    virtual ~IScheduler() {}

    /*! called by Manager on initialization.
     * The Scheduler cannot hold a reference to the Manager since the type is not known,
     * so all needed functions are passed as function objects
     */
    virtual void init_mgr_callbacks(
        std::shared_ptr< redGrapes::SchedulingGraph< TaskID, TaskPtr > > scheduling_graph,
        std::function< bool ( TaskPtr ) > run_task,
        std::function< void ( TaskPtr ) > activate_followers,
        std::function< void ( TaskPtr ) > remove_task
    ) = 0;

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

template <
    typename TaskID,
    typename TaskPtr
>
struct SchedulerBase : IScheduler< TaskID, TaskPtr >
{
    void init_mgr_callbacks(
        std::shared_ptr< redGrapes::SchedulingGraph< TaskID, TaskPtr > > scheduling_graph,
        std::function< bool ( TaskPtr ) > run_task,
        std::function< void ( TaskPtr ) > activate_followers,
        std::function< void ( TaskPtr ) > remove_task
    )
    {
        this->scheduling_graph = scheduling_graph;
        this->run_task = run_task;
        this->activate_followers = activate_followers;
        this->remove_task = remove_task;
    }

protected:
    std::shared_ptr< redGrapes::SchedulingGraph< TaskID, TaskPtr > > scheduling_graph;
    std::function< bool ( TaskPtr ) > run_task;
    std::function< void ( TaskPtr ) > activate_followers;
    std::function< void ( TaskPtr ) > remove_task;
};

} // namespace scheduler

} // namespace redGrapes

