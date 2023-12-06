/* Copyright 2021-2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <redGrapes/dispatch/thread/cpuset.hpp>
#include <redGrapes/memory/allocator.hpp>
#include <redGrapes/task/queue.hpp>
// #include <redGrapes/task/task.hpp>
#include <redGrapes/memory/refcounted.hpp>

#include <atomic>
#include <shared_mutex>
#include <vector>

namespace redGrapes
{
    struct Task;

    struct TaskSpace;

    struct TaskSpaceDeleter
    {
        void operator()(TaskSpace* space)
        {
            delete space;
        }
    };

    /*! TaskSpace handles sub-taskspaces of child tasks
     */
    struct TaskSpace : memory::Refcounted<TaskSpace, TaskSpaceDeleter>
    {
        std::vector<memory::Refcounted<TaskSpace, TaskSpaceDeleter>::Guard> active_child_spaces;
        std::shared_mutex active_child_spaces_mutex;

        std::atomic<unsigned long> task_count;
        Task* parent;
        unsigned depth;

        virtual ~TaskSpace();

        // top space
        TaskSpace();

        // sub space
        TaskSpace(Task* parent);

        virtual bool is_serial(Task& a, Task& b);
        virtual bool is_superset(Task& a, Task& b);

        // add a new task to the task-space
        void submit(Task* task);

        // remove task from task-space
        void free_task(Task* task);

        bool empty() const;
    };

} // namespace redGrapes
