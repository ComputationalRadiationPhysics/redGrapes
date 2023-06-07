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

using WakerID = int16_t;

/*! Scheduler Interface
 */
struct IScheduler
{
    virtual ~IScheduler()
    {
    }

    virtual void start()
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

    //! give worker work if available
    virtual Task * schedule( dispatch::thread::WorkerThread & worker )
    {
        return nullptr;
    }

    virtual bool wake( WakerID id = 0 )
    {
        return false;
    }
    
    virtual void wake_all_workers()
    {}

    virtual bool wake_one_worker()
    {
        return false;
    }
};

} // namespace scheduler

} // namespace redGrapes

