/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <functional>
#include <memory>

namespace redGrapes
{

struct Task;

namespace scheduler
{

/*! Scheduler Interface
 */
struct IScheduler
{
    virtual ~IScheduler() {}

    virtual std::shared_ptr<Task> get_job()
    {
        return std::shared_ptr<Task>();
    }
    
    /*! whats the task dependency type for the edge a -> b (task a precedes task b)
     * @return true if task b depends on the pre event of task a, false if task b depends on the post event of task b.
     */
    virtual bool task_dependency_type( std::shared_ptr<Task> a, std::shared_ptr<Task> b )
    {
        return false;
    }

    virtual void activate_task( std::shared_ptr<Task> task_vertex ) {}

    //! wakeup to call activate_next()
    virtual void notify() {}
};

} // namespace scheduler

} // namespace redGrapes

