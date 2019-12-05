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
    using typename SchedulerBase<TaskID, TaskPtr, PrecedenceGraph>::Job;

    struct TaskProperties
    {
        TaskState state;
    };

    StateScheduler( std::shared_ptr<PrecedenceGraph> & pg, size_t n_threads )
        : SchedulerBase< TaskID, TaskPtr, PrecedenceGraph >( pg, n_threads )
    {}

    TaskState get_task_state( TaskID task_id )
    {
        std::lock_guard<std::mutex> l(mutex);
        return states[task_id];
    }

    bool is_task_ready( TaskPtr task_ptr )
    {
        return boost::in_degree(task_ptr.vertex, task_ptr.graph->graph()) == 0;
    }

    template <typename Task>
    auto add_task( Task task, std::shared_ptr<PrecedenceGraph> g )
    {
        auto l = g->unique_lock();
        auto vertex = g->push( task );
        TaskPtr task_ptr{ g, vertex };
        this->scheduling_graph.add_task( task_ptr );
        bool ready = is_task_ready( task_ptr );
        l.unlock();

        task.hook_before([this, task_id=task.task_id]
                         {
                             std::unique_lock<std::mutex> l( mutex );
                             states[ task_id ] = TaskState::running;
                         });
        task.hook_after([this, task_id=task.task_id]
                        {
                            std::unique_lock<std::mutex> l( mutex );
                            states[ task_id ] = TaskState::done;
                            l.unlock();
                            this->notify();
                        });

        {
            std::unique_lock<std::mutex> l( mutex );
            states[ task.task_id ] = TaskState::pending;
            if( ready )
                active_tasks.push_back( task_ptr );
        }

        this->notify();

        return task_ptr;
    }

    void update_vertex( TaskPtr p )
    {
        auto vertices = this->scheduling_graph.update_vertex( p );

        for( auto v : vertices )
        {
            TaskPtr following_task{ p.graph, v };
            if( is_task_ready( following_task ) )
            {
                std::unique_lock<std::mutex> tasks_lock( mutex );
                active_tasks.push_back( following_task );
            }
        }
        this->notify();
    }

    /*
     * update task states and remove done tasks
     * @return ready tasks
     */
    std::vector<Job> update_graph()
    {
        std::unique_lock<std::mutex> tasks_lock( mutex );
        std::vector< Job > new_jobs;

        for(int i = 0; i < active_tasks.size(); ++i)
        {
            auto task_ptr = active_tasks[i];

            auto l = task_ptr.graph->unique_lock();
            auto task_id = task_ptr.get().task_id;

            switch( states[task_id] )
            {
            case TaskState::done:
                if( this->scheduling_graph.is_task_finished( task_id ) )
                {
                    std::vector<TaskPtr> potential_ready;
                    {
                        for( auto edge_it = boost::out_edges(task_ptr.vertex, task_ptr.graph->graph()); edge_it.first != edge_it.second; ++edge_it.first)
                        {
                            auto target_vertex = boost::target(*edge_it.first, task_ptr.graph->graph());
                            if( boost::in_degree(target_vertex, task_ptr.graph->graph()) == 1 )
                            {
                                TaskPtr p{ task_ptr.graph, target_vertex };
                                active_tasks.push_back( p );
                            }
                        }
                    }

                    task_ptr.graph->finish( task_ptr.vertex );
                    active_tasks.erase( active_tasks.begin() + i );
                    --i;

                    states.erase( task_id );
                }
                break;

            case TaskState::pending:
                {
                    if( is_task_ready( task_ptr ) )
                    {
                        states[ task_id ] = TaskState::ready;
                        new_jobs.push_back( Job{ task_ptr.get().impl, task_ptr } );
                    }
                }
                break;
            }
        }

        return new_jobs;
    }

private:
    std::mutex mutex;
    std::unordered_map< TaskID, TaskState > states;
    std::vector< TaskPtr > active_tasks; // contains ready, running & done tasks
};

} // namespace redGrapes
