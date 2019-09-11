
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
    {
        graph.set_notify_hook( [this]{ this->notify(); } );
        for( auto & t : this->graph.schedule )
            t.set_request_hook( [this]{ this->notify(); } );
    }

    void notify()
    {
        auto l = this->graph.precedence_graph.lock();
        //std::cout << "scheduler notify()" << std::endl;

        for( Task * task : this->collect_tasks(
                                               [this](Task * task)
                                               {
                                                   //std::cerr << "test " << task;
                                                   if( this->states.count(task) )
                                                   {
                                                       //std::cerr << "has state..";
                                                       //std::cerr << ".. has state " << this->states[task] << std::endl;
                                                       return this->states[ task ] == TaskState::done;
                                                   }
                                                   else
                                                   {
                                                       //std::cerr << ".. has no state yet"<<std::endl;
                                                       return false;
                                                   }
                                               })
             )
        {
            //std::cerr << "TASK " << task << "done: ";
            if( this->graph.precedence_graph.finish( task ) )
            {
                //std::cerr << "destroy task" <<std::endl;
                states.erase( task );
                delete task;
            }
            //else
                //std::cerr << "task not yet removed" << std::endl;
        }

        for( auto it = this->graph.precedence_graph.vertices(); it.first!=it.second; ++it.first )
        {
            auto task = *(it.first);
            if( states.count(task) == 0 )
            {
                // new task
                states.prepare_task_states( task );
                this->graph.make_job( task );
            }

            if( states[task] == TaskState::pending && this->is_task_ready( task ) )
                states[task] = TaskState::ready;
        }

        for( auto & thread : this->graph.schedule )
        {
            if( thread.needs_job() )
                schedule( thread );
        }
    }

    void schedule( typename SchedulingGraph::ThreadSchedule & thread )
    {
        //std::cout << "thread needs task" << std::endl;
        if( std::experimental::optional<Task *> task =
                this->find_task(
                    [this]( Task * task )
                    {
                        return this->states[ task ] == TaskState::ready;
                    }
                )
        )
        {
            //std::cerr << "task["<<*task<<"] has state = "<<states[*task]<<", set state to "<<TaskState::scheduled<<std::endl;
            states[ *task ] = TaskState::scheduled;
            //std::cerr << "fifo: Found Task " << *task <<std::endl;
            thread.push( typename SchedulingGraph::Job{ *task } );
        }
    }
};

} // namespace rmngr

