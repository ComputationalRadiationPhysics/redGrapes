/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <optional>
#include <spdlog/spdlog.h>

#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/context.hpp>

namespace redGrapes
{
namespace dispatch
{
namespace thread
{

void execute_task( Task & task )
{
    assert( task.is_ready() );
    SPDLOG_TRACE("thread dispatch: execute task {}", task.task_id);

    task.get_pre_event().notify();
    current_task = &task;

    if( auto event = task() )
    {
        task.sg_pause( *event );

        task.pre_event.up();
        task.get_pre_event().notify();
    }
    else
    {
        auto e = task.get_post_event();
        e.notify();
    }

    current_task = nullptr;
}

} // namespace thread
} // namespace dispatch
} // namespace redGrapes

