/* Copyright 2021-2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/dispatch/thread/worker.hpp>
#include <redGrapes/memory/allocator.hpp>
#include <redGrapes/memory/block.hpp>
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/task/queue.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/util/trace.hpp>

namespace redGrapes
{
    void TaskSpaceDeleter::operator()(TaskSpace* s)
    {
        delete s;
    }

    TaskSpace::~TaskSpace()
    {
    }

    TaskSpace::TaskSpace() : depth(0), parent(nullptr)
    {
        task_count = 0;
    }

    // sub space
    TaskSpace::TaskSpace(Task* parent) : depth(parent->space->depth + 1), parent(parent)
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
        return tc == 0;
    }

    void TaskSpace::free_task(Task* task)
    {
        TRACE_EVENT("TaskSpace", "free_task()");
        unsigned count = task_count.fetch_sub(1) - 1;

        unsigned arena_id = task->arena_id;
        task->~Task();

        // FIXME: len of the Block is not correct since FunTask object is bigger than sizeof(Task)
        SingletonContext::get().worker_pool->get_worker(arena_id).alloc.deallocate(
            memory::Block{(uintptr_t) task, sizeof(Task)});

        // TODO: implement this using post-event of root-task?
        //  - event already has in_edge count
        //  -> never have current_task = nullptr
        // spdlog::info("kill task... {} remaining", count);
        if(count == 0)
            SingletonContext::get().scheduler->wake_all();
    }

    void TaskSpace::submit(Task* task)
    {
        TRACE_EVENT("TaskSpace", "submit()");
        task->space.acquire(this);
        task->task = task;

        ++task_count;

        if(parent)
            assert(this->is_superset(*parent, *task));

        for(auto r = task->unique_resources.rbegin(); r != task->unique_resources.rend(); ++r)
        {
            r->task_entry = r->resource->users.push(task);
        }

        SingletonContext::get().scheduler->emplace_task(*task);
    }

} // namespace redGrapes
