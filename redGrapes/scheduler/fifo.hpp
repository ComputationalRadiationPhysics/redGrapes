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
    enum TaskState
    {
        uninitialized = 0,
        pending,
        ready,
        running,
        paused,
        done
    };

    struct TaskProperties
    {
        TaskState state;
    };

    std::recursive_mutex mutex;
    std::unordered_map< TaskID, TaskState > states;

    //! contains ready, running and paused tasks
    std::vector< TaskPtr > active_tasks;
    
    //! contains ready tasks
    std::queue< TaskPtr > task_queue;

    /*
     * manager callbacks
     */
    std::function< bool ( TaskPtr ) > run_task;
    std::function< void ( TaskPtr ) > remove_task;

    std::shared_ptr< PrecedenceGraph > precedence_graph;
    redGrapes::SchedulingGraph< TaskID, TaskPtr > & scheduling_graph;

    FIFO(
        std::shared_ptr< PrecedenceGraph > precedence_graph,
        redGrapes::SchedulingGraph<TaskID, TaskPtr> & scheduling_graph,
        std::function< bool ( TaskPtr ) > mgr_run_task,
        std::function< void ( TaskPtr ) > mgr_finish_task
    ) :
        precedence_graph( precedence_graph ),
        scheduling_graph( scheduling_graph ),
        run_task( mgr_run_task ),
        remove_task( mgr_finish_task )
    {}

    std::optional< TaskPtr > get_job()
    {
        std::unique_lock< std::recursive_mutex > l( mutex );

        if( task_queue.empty() )
        {
            //l.unlock();
            update( l );
            //l.lock();
        }

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
        if( auto task_ptr = get_job() )
        {
            auto task_id = task_ptr->locked_get().task_id;

            {
                std::unique_lock< std::recursive_mutex > l( mutex );
                states[ task_id ] = running;
            }

            bool finished = run_task( *task_ptr );

            {
                std::unique_lock< std::recursive_mutex > l( mutex );
                states[ task_id ] = finished ? done : paused;

                if( finished )
                    scheduling_graph.task_end( task_id );
            }

            return true;
        }
        else
            return false;
    }

    //! update all active tasks
    void update( std::unique_lock< std::recursive_mutex > & l )
    {
        //( mutex );

        for( int i = 0; i < active_tasks.size(); ++i )
        {
            auto task_ptr = active_tasks[ i ];
            auto task_id = task_ptr.locked_get().task_id;

            switch( states[ task_id ] )
            {
            case TaskState::done:
                /* if there are there events which must precede the tasks post-event
                 * we can not remove the task yet.
                 */
                if( scheduling_graph.is_task_finished( task_id ) )
                {
                    // remove task from active_tasks
                    active_tasks.erase( active_tasks.begin() + i );
                    -- i;

                    l.unlock();

                    // remove task from both graphs and activate its followers
                    remove_task( task_ptr );

                    l.lock();
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
        }
    }

    void activate_task( TaskPtr task_ptr )
    {
        std::unique_lock< std::recursive_mutex > l( mutex );
        auto task_id = task_ptr.locked_get().task_id;

        if( ! scheduling_graph.is_task_finished( task_id ) )
        {
            if( ! states.count( task_id ) ) // || states[ task_id ] = uninitialized
            {
                states[ task_id ] = pending;
                active_tasks.push_back( task_ptr );
            }

            switch( states[ task_id ] )
            {
            case TaskState::paused:
            case TaskState::pending:                
                if( scheduling_graph.is_task_ready( task_id ) )
                {
                    states[ task_id ] = ready;
                    task_queue.push( task_ptr );
                }            
            }
        }
    }
};

} // namespace scheduler

} // namespace redGrapes

