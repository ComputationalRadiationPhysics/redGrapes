/* Copyright 2021-2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/util/trace.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/task/queue.hpp>
#include <redGrapes/util/allocator.hpp>

namespace redGrapes
{
    TaskSpace::~TaskSpace()
    {
    }

    TaskSpace::TaskSpace()
        : depth(0)
        , parent(nullptr)
    {
        task_count = 0;
    }

    // sub space
    TaskSpace::TaskSpace(Task * parent)
        : depth(parent->space->depth + 1)
        , parent(parent)
    {
        task_count = 0;
    }

    bool TaskSpace::is_serial(Task& a, Task& b)
    {
        return ResourceUser::is_serial(a, b);
    }

    bool TaskSpace::is_superset(Task& a, Task& b)
    {
        return ResourceUser::is_superset(a, b);
    }

    bool TaskSpace::empty() const
    {
        unsigned tc = task_count.load();
        SPDLOG_TRACE("({}) empty? task count = {}", (void*)this, tc);
        return tc == 0;
    }

    void TaskSpace::free_task( Task * task )
    {
        unsigned count = task_count.fetch_sub(1) - 1;

        unsigned arena_id = task->arena_id;
        task->~Task();
        memory::alloc->deallocate(arena_id, &task);

        // TODO: implement this using post-event of root-task?
        //  - event already has in_edge count
        //  -> never have current_task = nullptr
        //spdlog::info("kill task... {} remaining", count);
        if( count == 0 )
        {
            //spdlog::info("last task, wake all");
            top_scheduler->wake_all_workers();
        }
    }

    void TaskSpace::submit( Task * task )
    {
        task->space = shared_from_this();
        task->task = task;

        ++ task_count;

        if( parent )
            assert( this->is_superset(*parent, *task) );

        for( ResourceEntry & r : task->unique_resources )
            r.task_idx = r.resource->users.push( task );

        top_scheduler->emplace_task( *task );
    }

} // namespace redGrapes
