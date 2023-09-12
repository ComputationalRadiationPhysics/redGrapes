/* Copyright 2022-2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <optional>
#include <spdlog/spdlog.h>

#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/context.hpp>
#include <redGrapes/util/trace.hpp>

namespace redGrapes
{
namespace dispatch
{
namespace thread
{

thread_local scheduler::WakerId current_waker_id;
thread_local std::shared_ptr< WorkerThread > current_worker;

void execute_task( Task & task )
{
    TRACE_EVENT("Worker", "dispatch task");

    SPDLOG_DEBUG("thread dispatch: execute task {}", task.task_id);
    assert( task.is_ready() );

    task.get_pre_event().notify();
    current_task = &task;

    auto event = task();
    
    if( event )
    {
        event->get_event().waker_id = current_waker_id;
        task.sg_pause( *event );

        task.pre_event.up();
        task.get_pre_event().notify();
    }
    else
        task.get_post_event().notify();

    current_task = nullptr;
}

} // namespace thread
} // namespace dispatch
} // namespace redGrapes

