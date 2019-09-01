
#pragma once

#include <mutex>
#include <akrzemi/optional.hpp>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/states.hpp>

namespace rmngr
{

template < typename SchedulingGraph >
struct FIFOScheduler
    : SchedulerBase< SchedulingGraph >
{
    using typename SchedulerBase< SchedulingGraph >::Task;

    TaskStateMap< Task* > states;

    FIFOScheduler( SchedulingGraph & graph )
        : SchedulerBase< SchedulingGraph >( graph )
    {}

    void notify()
    {
        for( auto & thread : this->graph.schedule )
        {
            if( thread.empty() )
                schedule( thread );
            else if( auto j = thread.get_current_job() )
                if( states[(*j).task] == TaskState::done )
                    schedule( thread );
        }
    }

    void schedule( typename SchedulingGraph::ThreadSchedule & thread )
    {
        std::unique_lock<std::mutex> lock(this->graph.mutex);

        this->remove_tasks( [this](Task * task){ return states[ task ] == TaskState::done; } );

        if( std::experimental::optional<Task *> task = this->find_task(
                [this]( Task * task )
                {
                    return
                        states[ task ] == TaskState::pending &&
                        this->is_task_ready( task );
                } ))
        {
            states.prepare_task_states( *task );
            (*task)->hook_after( [this]{ notify(); } );

            lock.unlock();
            thread.push( this->graph.make_job( *task ) );
        }
    }
};

} // namespace rmngr

