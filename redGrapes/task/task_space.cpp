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

    /*! take tasks from the emplacement queue and initialize them,
     *  until a task is initialized whose execution could start immediately
     *
     * @return false if queue is empty
     */
    bool TaskSpace::init_dependencies( )
    {
        Task * t;
        return init_dependencies(t, false);
    }

    bool TaskSpace::init_dependencies( Task* & t, bool claimed )
    {
        SPDLOG_TRACE("TaskSpace::init_until_ready() this={}", (void*)this);

        if(Task * task = emplacement_queue.pop())
        {
            task->alive = 1;
            task->pre_event.up();
            task->init_graph();

            if( task->get_pre_event().notify( claimed ) )
                t = task;

            return true;
        }
        else
        {
            SPDLOG_TRACE("TaskSpace::init_until_ready(): check child spaces");
            std::shared_lock< std::shared_mutex > read_lock( active_child_spaces_mutex );
            for( auto child : active_child_spaces )
                if( child->init_dependencies( t, claimed ) )
                    return true;

            return false;
        }
    }

    void TaskSpace::kill(Task &task) {
      if (__sync_bool_compare_and_swap(&task.alive, 1, 0)) {
        if (task.children) {
          SPDLOG_TRACE("remove child space");

          std::unique_lock<std::shared_mutex> wr_lock(
              current_task->space->active_child_spaces_mutex);
          active_child_spaces.erase(std::find(active_child_spaces.begin(),
                                              active_child_spaces.end(),
                                              task.children));
        }

        SPDLOG_TRACE("remove task {}", task.task_id);
        task.delete_from_resources();
        task.~Task();
        task_storage.deallocate(&task);

        auto ts = top_scheduler;

        if (task_count.fetch_sub(1) == 1) {
          SPDLOG_DEBUG("task space empty");

          if (parent)
            parent->space->try_remove(*parent);

          if (ts)
            ts->wake_all_workers();
        }

        SPDLOG_TRACE("kill: task count = {}", task_count);
      }
    }

    void TaskSpace::try_remove(Task& task)
    {
        SPDLOG_TRACE("try remove {}", task.task_id);
        if( task.post_event.is_reached() )
            if( task.result_get_event.is_reached() )
                if( !task.children || task.children->empty() )
                    kill(task);
                else
                    SPDLOG_TRACE("task {} not yet removed: has children", task.task_id);
            else
                SPDLOG_TRACE("task {} not yet removed: result not taken", task.task_id);
        else
            SPDLOG_TRACE("task {} not yet removed: post event not reached (still running)", task.task_id);

    }

    bool TaskSpace::empty() const
    {
        unsigned tc = task_count.load();
        SPDLOG_TRACE("({}) empty? task count = {}", (void*)this, tc);
        return tc == 0;
    }

} // namespace redGrapes
