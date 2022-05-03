/* Copyright 2019-2021 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <unordered_set>
#include <optional>
#include <atomic>

#include <moodycamel/concurrentqueue.h>

#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/dispatch/thread/worker.hpp>
#include <redGrapes/task/task.hpp>

#include <redGrapes/redGrapes.hpp>

namespace redGrapes
{
namespace scheduler
{

struct FIFO : public IScheduler
{
    moodycamel::ConcurrentQueue< std::shared_ptr<Task> > ready;

    void activate_task( std::shared_ptr<Task> task )
    {
        SPDLOG_TRACE("FIFO: activate task {}", task->task_id);
        ready.enqueue(task);
    }

    /*! take a job from the ready queue
     * if none available, update 
     */
    std::shared_ptr<Task> get_job()
    {
        if( auto task_vertex = try_next_task() )
            return task_vertex;
        else
        {
            update_active_task_spaces();
            return try_next_task();
        }
    }

    /*! call the manager to activate tasks until we get at least
     * one in the ready queue
     */
    std::shared_ptr<Task> try_next_task()
    {
        std::shared_ptr<Task> task;
        if(ready.try_dequeue(task))
            return task;

        return std::shared_ptr<Task>();
    }
};

} // namespace scheduler
} // namespace redGrapes

