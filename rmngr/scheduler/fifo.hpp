
#pragma once

#include <mutex>
#include <queue>
#include <akrzemi/optional.hpp>
#include <rmngr/scheduler/scheduler.hpp>

namespace rmngr
{

template < typename SchedulingGraph >
struct FIFOScheduler
    : SchedulerBase< SchedulingGraph >
{
    using typename SchedulerBase< SchedulingGraph >::Task;
    using Job = typename SchedulingGraph::Job;

    enum TaskState { uninitialized, pending, ready, running, done };

    std::mutex states_mutex;
    std::unordered_map< Task*, TaskState > states;

    std::mutex queue_mutex;
    std::queue< Job > job_queue;

    FIFOScheduler( SchedulingGraph & graph )
        : SchedulerBase< SchedulingGraph >( graph )
    {
        for( auto & t : this->graph.schedule )
            t.set_request_hook( [this,&t]{ get_job(t); } );
    }

    void new_task( Task * task )
    {
        task->hook_before( [this, task]
            {
                std::lock_guard<std::mutex> lock(this->states_mutex);
                states[ task ] = TaskState::running;
            });

        task->hook_after( [this, task]
            {
                std::lock_guard<std::mutex> lock(this->states_mutex);
                states[ task ] = TaskState::done;

                this->uptodate.clear();
            });

        this->graph.make_job( task );

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
        if( thread.needs_job() )
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
    }

    void update_graph()
    {
        std::lock_guard<std::mutex> lock(this->states_mutex);

        auto l = this->graph.precedence_graph.lock();

        for( auto task : this->collect_tasks( [this](Task * task){ return this->states[ task ] == TaskState::done; }) )
        {
            if( this->graph.precedence_graph.finish( task ) )
            {
                states.erase( task );
                //delete task;
            }
        }

        for( auto task : this->collect_tasks( [this](Task * task){ return this->states[ task ] == TaskState::pending; }) )
        {
            if( this->is_task_ready( task ) )
            {
                states[task] = TaskState::ready;
                job_queue.push( Job{ task } );
            }
        }
    }
};

} // namespace rmngr

