

#pragma once

#include <vector>
#include <rmngr/task/task_container.hpp>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/thread/thread_dispatcher.hpp>

namespace rmngr
{

enum TaskState { uninitialized = 0, pending, ready, running, done };

template <
    typename TaskProperties,
    typename SchedulingGraph
>
struct StateScheduler
    : SchedulerBase< TaskProperties, SchedulingGraph >
{
    using TaskID = typename TaskContainer<TaskProperties>::TaskID;

    StateScheduler( TaskContainer< TaskProperties > & tasks, SchedulingGraph & graph )
        : SchedulerBase< TaskProperties, SchedulingGraph >( tasks, graph )
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

    template <typename Refinement>
    void push( TaskID task, Refinement & ref )
    {
        this->tasks.task_hook_before(
            task,
            [this, task]
            {
                this->set_task_state( task, TaskState::running );
            });

        this->tasks.task_hook_after(
            task,
            [this, task]
            {
                this->set_task_state( task, TaskState::done );
            });

        this->graph.add_task( task, ref );

        this->set_task_state( task, TaskState::pending );
        this->notify();
    }

    /*
     * update task states and remove done tasks
     * @return ready tasks
     */
    std::vector<TaskID> update_graph()
    {
        std::lock_guard<std::mutex> lock(this->states_mutex);
        for( auto task : this->collect_tasks( [this](TaskID task){ return states[task] == TaskState::done; }) )
        {
            if( this->graph.precedence_graph.finish( task ) )
            {
                states.erase( task );
                this->tasks.erase( task );
            }
        }

        std::vector< TaskID > ready;
        for( auto task : this->collect_tasks( [this](TaskID task){ return states[task] == TaskState::pending; }) )
        {
            if( this->is_task_ready( task ) )
            {
                states[task] = TaskState::ready;
                ready.push_back( task );
            }
        }

        return ready;
    }

private:
    std::mutex states_mutex;
    std::unordered_map< TaskID, TaskState > states;    
};

} // namespace rmngr

