/* Copyright 2019-2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <optional>
#include <moodycamel/concurrentqueue.h>

#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/redGrapes.hpp>

namespace redGrapes
{
namespace scheduler
{

struct WorkerQueue
{
    Task volatile * volatile ready;
    //    Task volatile * volatile pending;    
};

struct FIFO : public IScheduler
{
    FIFO(std::vector<>workers)
    {

        
    }
    
    void activate_task( Task & task )
    {
        SPDLOG_TRACE("FIFO: activate task {}", task.task_id);
        ready.enqueue(&task);
    }

    void schedule()
    {
        SPDLOG_TRACE("balance workers");
        update_active_task_spaces();
    }

    /*! take a job from the ready queue
     * if none available, update 
     */
    Task* get_job()
    {
        if( auto task = try_next_task() )
            return task;
        else
        {
            update_active_task_spaces();
            return try_next_task();
        }
    }

    /*! call the manager to activate tasks until we get at least
     * one in the ready queue
     */
    Task* try_next_task()
    {
        Task* task;
        if(ready.try_dequeue(task))
            return task;

        return nullptr;
    }
};

} // namespace scheduler
} // namespace redGrapes

