
#pragma once

#include <mutex>
#include <queue>
#include <akrzemi/optional.hpp>
#include <rmngr/scheduler/scheduler.hpp>

namespace rmngr
{

template <
    typename TaskProperties,
    typename SchedulingGraph
>
struct FIFOScheduler
    : SchedulerBase< SchedulingGraph >
{
    using TaskID = typename TaskContainer<TaskProperties>::TaskID;
    using Job = typename SchedulingGraph::Job;

    enum TaskState { uninitialized, pending, ready, running, done };

    std::mutex states_mutex;
    std::unordered_map< TaskID, TaskState > states;

    std::mutex queue_mutex;
    std::queue< Job > job_queue;

    TaskContainer< TaskProperties > & tasks;

    FIFOScheduler( TaskContainer< TaskProperties > & tasks, SchedulingGraph & graph )
        : SchedulerBase< SchedulingGraph >( graph )
        , tasks( tasks )
    {
        for( auto & t : this->graph.schedule )
            t.set_request_hook( [this,&t]{ get_job(t); } );
    }

    void push( TaskID task )
    {
        tasks.task_hook_before(
            task,
            [this, task]
            {
                std::lock_guard<std::mutex> lock(this->states_mutex);
                states[ task ] = TaskState::running;
            });

        tasks.task_hook_after(
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

private:
    std::atomic_flag uptodate;

    void get_job( typename SchedulingGraph::ThreadSchedule & thread )
    {
        std::unique_lock< std::mutex > lock( queue_mutex );

        if( job_queue.empty() )
        {
            bool u1 = this->uptodate.test_and_set();
            bool u2 = this->graph.precedence_graph.test_and_set();
            if( !u1 || !u2 )
                update_graph();
        }

        if( ! job_queue.empty() )
        {
            auto job = job_queue.front();
            job_queue.pop();
            thread.push( job );
        }
    }

    void update_graph()
    {
        auto l = this->graph.precedence_graph.lock();
        std::lock_guard<std::mutex> lock(this->states_mutex);

        for( auto task : this->collect_tasks( [this](TaskID task){ return this->states[ task ] == TaskState::done; }) )
        {
            if( this->graph.precedence_graph.finish( task ) )
            {
                states.erase( task );
                //tasks.erase( task );
            }
        }

        for( auto task : this->collect_tasks( [this](TaskID task){ return this->states[ task ] == TaskState::pending; }) )
        {
            if( this->is_task_ready( task ) )
            {
                states[task] = TaskState::ready;
                job_queue.push( Job{ tasks, task } );
            }
        }
    }
};

} // namespace rmngr

