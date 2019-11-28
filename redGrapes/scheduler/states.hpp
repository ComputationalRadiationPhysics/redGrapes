/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <vector>
#include <redGrapes/task/task.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/thread/thread_dispatcher.hpp>

namespace redGrapes
{

enum TaskState { uninitialized = 0, pending, ready, running, done };

template <
    typename TaskID,
    typename TaskPtr,
    typename PrecedenceGraph
>
struct StateScheduler
    : SchedulerBase< TaskID, TaskPtr, PrecedenceGraph >
{
    using Job = typename SchedulingGraph<TaskID, TaskPtr>::Job;

    StateScheduler( std::unordered_map<TaskID, TaskPtr> & tasks, std::shared_mutex & m, std::shared_ptr<PrecedenceGraph> & pg, size_t n_threads )
        : SchedulerBase< TaskID, TaskPtr, PrecedenceGraph >( pg, n_threads )
        , tasks( tasks )
        , tasks_mutex( m )
    {}

    void set_task_state( TaskID id, TaskState state )
    {
        {
            std::lock_guard<std::mutex> lock(this->states_mutex);
            states[ id ] = state;
        }
        this->uptodate.clear();
    }

    TaskState get_task_state( TaskID id )
    {
        std::lock_guard<std::mutex> lock(this->states_mutex);
        return states[ id ];
    }

    template <typename Task, typename Graph>
    auto add_task( Task && task, Graph & g )
    {
        auto task_id = task.task_id;
        task.hook_before([this, task_id]{ this->set_task_state( task_id, TaskState::running ); });
        task.hook_after([this, task_id]{ this->set_task_state( task_id, TaskState::done ); });
        auto v = this->scheduling_graph.add_task( std::move(task), g );

        this->set_task_state( task_id, TaskState::pending );
        this->notify();

        return v;
    }

    /*
     * update task states and remove done tasks
     * @return ready tasks
     */
    std::vector<Job> update_graph()
    {
        std::lock_guard<std::mutex> lock(this->states_mutex);

        std::vector<TaskID> sel;

        this->precedence_graph->template collect_vertices<TaskID>(
            sel,
            [this](auto const & task) -> std::experimental::optional<TaskID> {
                if( states[task.task_id] == TaskState::done )
                    return task.task_id;
                else
                    return std::experimental::nullopt;
            });

        for( TaskID task_id : sel )
        {
            if( this->scheduling_graph.is_task_finished( task_id ) )
            {
                TaskPtr p;
                {
                    std::shared_lock<std::shared_mutex> lock( tasks_mutex );
                    p = this->tasks[task_id];
                }
                p.graph->finish( p.vertex );
                states.erase( task_id );

                {
                    std::unique_lock<std::shared_mutex> lock( tasks_mutex );
                    this->tasks.erase( task_id );
                }
            }
        }

        std::vector< Job > ready;
        this->precedence_graph->template collect_vertices<Job>(
            ready,
            [this](auto const & task) -> std::experimental::optional< Job > {
                if( states[task.task_id] == TaskState::pending )
                {
                    std::shared_lock<std::shared_mutex> lock( tasks_mutex );
                    if( this->tasks.count(task.task_id) )
                        if( this->is_task_ready(this->tasks[task.task_id]) )
                            return Job{ task.impl, task.task_id };
                }
                return std::experimental::nullopt;
            });

        for( auto job : ready )
            states[job.task_id] = TaskState::ready;

        return ready;
    }

protected:
    std::unordered_map<TaskID, TaskPtr> & tasks;
    std::shared_mutex & tasks_mutex;

private:
    std::mutex states_mutex;
    std::unordered_map< TaskID, TaskState > states;    
};

} // namespace redGrapes
