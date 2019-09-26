

#pragma once

#include <vector>
#include <rmngr/task/task_container.hpp>
#include <rmngr/scheduler/scheduler.hpp>

namespace rmngr
{

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

    void push( TaskID task )
    {
        this->tasks.task_hook_before(
            task,
            [this, task]
            {
                std::lock_guard<std::mutex> lock(this->states_mutex);
                states[ task ] = TaskState::running;
            });

        this->tasks.task_hook_after(
            task,
            [this, task]
            {
                std::lock_guard<std::mutex> lock(this->states_mutex);
                states[ task ] = TaskState::done;

                this->uptodate.clear();
            });

        this->graph.add_task( task );

        {
            std::lock_guard<std::mutex> lock(this->states_mutex);
            states[task] = TaskState::pending;
        }

        this->uptodate.clear();

        for( auto & t : this->graph.schedule )
            t.notify();
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
    enum TaskState { uninitialized, pending, ready, running, done };

    std::mutex states_mutex;
    std::unordered_map< TaskID, TaskState > states;    
};

} // namespace rmngr

