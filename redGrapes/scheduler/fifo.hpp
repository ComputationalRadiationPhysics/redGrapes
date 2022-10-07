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

struct FIFO : public IScheduler
{
    std::mutex m;
    task::Queue ready;

    void activate_task( Task & task )
    {
        SPDLOG_TRACE("FIFO: activate task {}", task.task_id);
        ready.push(&task);
    }

    /*! take a job from the ready queue
     * if none available, update 
     */
    Task* get_job()
    {
        return ready.pop();
    }
};

} // namespace scheduler
} // namespace redGrapes

