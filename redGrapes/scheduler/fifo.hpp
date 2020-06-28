/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <queue>
#include <optional>

#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/thread/thread_local.hpp>

namespace redGrapes
{
namespace scheduler
{

template <
    typename TaskID,
    typename TaskPtr,
    typename PrecedenceGraph
>
struct FIFO : IScheduler< TaskPtr >
{
    enum TaskState { uninitialized = 0, pending, ready, running, paused, done };
    struct TaskProperties
    {
        TaskState state;
    };

    std::shared_ptr< PrecedenceGraph > precedence_graph;
    redGrapes::SchedulingGraph< TaskID, TaskPtr > & scheduling_graph;

    std::recursive_mutex mutex;
    std::unordered_map< TaskID, TaskState > states;
    std::vector< TaskPtr > active_tasks; // contains ready, running & done tasks

    std::queue< TaskPtr > task_queue;

    std::function< bool ( TaskPtr ) > run_task;
    std::function< void ( TaskPtr ) > finish_task;

    FIFO(
        std::shared_ptr< PrecedenceGraph > precedence_graph,
        redGrapes::SchedulingGraph<TaskID, TaskPtr> & scheduling_graph,
        std::function< bool ( TaskPtr ) > mgr_run_task,
        std::function< void ( TaskPtr ) > mgr_finish_task
    ) :
        precedence_graph( precedence_graph ),
        scheduling_graph( scheduling_graph ),
        run_task( mgr_run_task ),
        finish_task( mgr_finish_task )
    {}

    std::optional< TaskPtr > get_job()
    {
        std::lock_guard< std::recursive_mutex > l( mutex );

        if( task_queue.empty() )
            update();

        if( ! task_queue.empty() )
        {
            auto task_ptr = task_queue.front();
            task_queue.pop();

            return task_ptr;
        }
        else
            return std::nullopt;
    }

    bool consume()
    {
        std::unique_lock< std::recursive_mutex > l( mutex );

        if( auto task_ptr = get_job() )
        {
            auto task_id = task_ptr->locked_get().task_id;

            states[ task_id ] = running;

            l.unlock();
            bool finished = run_task( *task_ptr );
            l.lock();

            states[ task_id ] = finished ? done : paused;

            return true;
        }
        else
            return false;
    }

    //! update all active tasks
    void update()
    {
        std::lock_guard< std::recursive_mutex > lock( mutex );

        for( int i = 0; i < active_tasks.size(); ++i )
        {
            if( update_task( active_tasks[i] ) )
            {
                active_tasks.erase( active_tasks.begin() + i );
                -- i;
            }
        }
    }

    /*!
     * @return true if task was removed
     */
    bool update_task( TaskPtr task_ptr )
    {
        std::lock_guard< std::recursive_mutex > l( mutex );
        auto task_id = task_ptr.locked_get().task_id;

        switch( states[ task_id ] )
        {
        case TaskState::done:
            if( scheduling_graph.is_task_finished( task_id ) )
            {
                // remove task from both graphs and activate its followers
                finish_task( task_ptr );
                return true;
            }
            break;

        case TaskState::paused:
        case TaskState::pending:
            if( scheduling_graph.is_task_ready( task_id ) )
            {
                states[ task_id ] = ready;
                task_queue.push( task_ptr );
            }
            break;
        }

        return false;
    }

    void activate_task( TaskPtr task_ptr )
    {
        std::lock_guard< std::recursive_mutex > l( mutex );
        auto task_id = task_ptr.locked_get().task_id;

        if( ! scheduling_graph.is_task_finished( task_id ) )
        {
            if( ! states.count( task_id ) ) // || states[ task_id ] = uninitialized
            {
                states[ task_id ] = pending;
                active_tasks.push_back( task_ptr );
            }

            update_task( task_ptr );
        }
    }

};

} // namespace scheduler

} // namespace redGrapes
