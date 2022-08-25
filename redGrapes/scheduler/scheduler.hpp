/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <optional>
#include <spdlog/spdlog.h>

namespace redGrapes
{

struct Task;

namespace dispatch
{
namespace thread
{
struct WorkerThread;
}
}

namespace scheduler
{


struct IWaker
{
    virtual ~IWaker() {}

    virtual bool wake()
    {
        return false;
    }
};
  
/*! Scheduler Interface
 */
struct IScheduler : virtual IWaker
{
    virtual ~IScheduler()
    {
    }

    virtual void stop()
    {
    }

    /*! whats the task dependency type for the edge a -> b (task a precedes task b)
     * @return true if task b depends on the pre event of task a, false if task b depends on the post event of task b.
     */
    virtual bool task_dependency_type( Task const & a, Task const & b )
    {
        return false;
    }

    //! add task to ready set
    virtual void activate_task( Task & task ) {}
    virtual void schedule() {}
    virtual void schedule( dispatch::thread::WorkerThread & worker ) {}

    virtual void wake_all_workers() {}
    virtual void wake_one_worker() {}
};

} // namespace scheduler

} // namespace redGrapes

