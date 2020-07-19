/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace redGrapes
{
namespace scheduler
{

template <
    typename TaskID,
    typename TaskPtr
>
struct IScheduler
{
    virtual ~IScheduler() {}

    virtual void init_mgr_callbacks(
        std::shared_ptr< redGrapes::SchedulingGraph< TaskID, TaskPtr > > scheduling_graph,
        std::function< bool ( TaskPtr ) > run_task,
        std::function< void ( TaskPtr ) > activate_followers,
        std::function< void ( TaskPtr ) > remove_task
    ) = 0;

    virtual bool task_dependency_type( TaskPtr a, TaskPtr b )
    {
        return false;
    }

    virtual void activate_task( TaskPtr task_ptr ) = 0;

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

