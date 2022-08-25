/* Copyright 2021-2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/task/task.hpp>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/task/queue.hpp>

namespace redGrapes
{
    TaskSpace::~TaskSpace()
    {
    }

    TaskSpace::TaskSpace()
        : depth(0)
        , parent(nullptr)
        , next_id(0)
    {
        task_count = 0;
        task_capacity = 512;
    }

    // sub space
    TaskSpace::TaskSpace(Task& parent)
        : depth(parent.space->depth + 1)
        , parent(&parent)
    {
        task_count = 0;
        task_capacity = 512;
    }

    bool TaskSpace::is_serial(Task& a, Task& b)
    {
        return ResourceUser::is_serial(a, b);
    }

    bool TaskSpace::is_superset(Task& a, Task& b)
    {
        return ResourceUser::is_superset(a, b);
    }

    /*! take tasks from the emplacement queue and initialize them,
     *  until a task is initialized whose execution could start immediately
     *
     * @return true if ready task found,
     *         false if no tasks available
     */
    bool TaskSpace::init_until_ready()
    {
        std::lock_guard<std::mutex> lock(emplacement_mutex);

        while( task_capacity.fetch_sub(1) > 0 )
            //while(true)
        {
            if(auto task = emplacement_queue.pop())
            {
                task->alive = 1;
                task->pre_event.up();
                task->init_graph();
                if(task->get_pre_event().notify())
                    return true;
            } else {
                return false;
            }
        }

        return false;
    }

    void TaskSpace::try_remove(Task& task)
    {
        if(task.post_event.is_reached() && task.result_get_event.is_reached() && (!task.children || task.children->empty()))
        {
            if( __sync_bool_compare_and_swap(&task.alive, 1, 0) )
            {
                task_capacity++;

                task.delete_from_resources();
                task.~Task();
                task_storage.m_free(&task);

                auto ts = top_scheduler;

                if( task_count.fetch_sub(1) == 1 )
                {
                    SPDLOG_DEBUG("task space empty");
                    if( ts )
                        ts->wake_all_workers();
                }
            }
        }

        // todo multiple chunks!
    }

    bool TaskSpace::empty() const
    {
        return task_count == 0;
    }

} // namespace redGrapes
