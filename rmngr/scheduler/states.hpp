

#pragma once

#include <vector>
#include <rmngr/task/task_container.hpp>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/thread/thread_dispatcher.hpp>

namespace rmngr
{

enum TaskState { uninitialized, pending, ready, running, done };
    
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
        std::lock_guard<std::mutex> lock(this->states_mutex);
        states[ id ] = state;

        this->uptodate.clear();
        this->notify();
    }

    TaskState get_task_state( TaskID id )
    {
        std::lock_guard<std::mutex> lock(this->states_mutex);
        return states[ id ];
    }

    void push( TaskID task )
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

        this->graph.add_task( task );
        this->set_task_state( task, TaskState::pending );
    }

    /*
     * update task states and remove done tasks
     * @return ready tasks
     */
    std::vector<TaskID> update_graph()
    {
        auto l = this->graph.precedence_graph.lock();
        std::lock_guard<std::mutex> lock(this->states_mutex);

        for( auto task : this->collect_tasks( [this](TaskID task){ return this->states[ task ] == TaskState::done; }) )
        {
            if( this->graph.precedence_graph.finish( task ) )
            {
                states.erase( task );
                this->tasks.erase( task );
            }
        }

        std::vector< TaskID > ready;
        for( auto task : this->collect_tasks( [this](TaskID task){ return this->states[ task ] == TaskState::pending; }) )
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

