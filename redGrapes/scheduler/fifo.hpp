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
#include <redGrapes/thread_local.hpp>

namespace redGrapes
{
namespace scheduler
{

template <
    typename TaskID,
    typename TaskPtr
>
struct FIFO : public SchedulerBase< TaskID, TaskPtr >
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

private:
    std::mutex mutex;
    std::unordered_map< TaskID, TaskState > states;

    //! contains activated, not yet removed tasks (ready, paused, running)
    std::vector< std::pair< TaskID, TaskPtr > > active_tasks;

    //! contains ready tasks that are queued for execution
    std::queue< TaskPtr > task_queue;

public:
    //! returns true if a job was consumed, false if queue is empty
    bool consume()
    {
        if( auto task_ptr = get_job() )
        {
            auto task_id = task_ptr->locked_get().task_id;
            this->scheduling_graph->task_start( task_id );

            {
                std::unique_lock< std::mutex > l( mutex );
                states[ task_id ] = running;
            }

            bool finished = this->run_task( *task_ptr );

            if( finished )
            {
                this->scheduling_graph->task_end( task_id );
                this->activate_followers( *task_ptr );
            }

            {
                std::unique_lock< std::mutex > l( mutex );
                states[ task_id ] = finished ? done : paused;
            }

            return true;
        }
        else
            return false;
    }

    // precedence graph must be locked
    void activate_task( TaskPtr task_ptr )
    {
        std::unique_lock< std::mutex > l( mutex );
        auto task_id = task_ptr.get().task_id;

        if( ! this->scheduling_graph->is_task_finished( task_id ) )
        {
            if( ! states.count( task_id ) ) // || states[ task_id ] = uninitialized
            {
                states[ task_id ] = pending;
                active_tasks.push_back( std::make_pair( task_id, task_ptr ) );
            }

            switch( states[ task_id ] )
            {
            case TaskState::paused:
            case TaskState::pending:
                if( this->scheduling_graph->is_task_ready( task_id ) )
                {
                    states[ task_id ] = ready;
                    task_queue.push( task_ptr );
                }
            }
        }
    }

private:
    std::optional< TaskPtr > get_job()
    {
        std::unique_lock< std::mutex > l( mutex );

        if( task_queue.empty() )
            update( l );

        if( ! task_queue.empty() )
        {
            auto task_ptr = task_queue.front();
            task_queue.pop();

            return task_ptr;
        }
        else
            return std::nullopt;
    }

    //! update all active tasks
    void update( std::unique_lock< std::mutex > & l )
    {
        for( int i = 0; i < active_tasks.size(); ++i )
        {
            auto task_id = active_tasks[ i ].first;
            auto task_ptr = active_tasks[ i ].second;

            switch( states[ task_id ] )
            {
            case TaskState::done:
                /* if there are there events which must precede the tasks post-event
                 * we can not remove the task yet.
                 */
                if( this->scheduling_graph->is_task_finished( task_id ) )
                {
                    active_tasks.erase( active_tasks.begin() + i );
                    -- i;

                    l.unlock();
                    this->remove_task( task_ptr );
                    l.lock();
                }
                break;

            case TaskState::paused:
            case TaskState::pending:
                if( this->scheduling_graph->is_task_ready( task_id ) )
                {
                    states[ task_id ] = ready;
                    task_queue.push( task_ptr );
                }
                break;
            }
        }
    }
};

/*! Factory function to easily create a fifo-scheduler object
 */
template <
    typename Manager
>
auto make_fifo_scheduler(
    Manager & m
)
{
    return std::make_shared<
               FIFO<
                   typename Manager::TaskID,
                   typename Manager::TaskPtr
               >
           >();
}


} // namespace scheduler

} // namespace redGrapes

