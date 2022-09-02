/* Copyright 2019-2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <optional>
#include <functional>
#include <moodycamel/concurrentqueue.h>

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/scheduler/default_scheduler.hpp>

namespace redGrapes
{

thread_local Task * current_task;
thread_local std::function<void()> idle;

moodycamel::ConcurrentQueue< std::shared_ptr<TaskSpace> > active_task_spaces;

std::shared_ptr< TaskSpace > top_space;
std::shared_ptr< scheduler::IScheduler > top_scheduler;

std::shared_ptr<TaskSpace> current_task_space()
{
    if( current_task )
    {
        if( ! current_task->children )
        {
            auto task_space = std::make_shared<TaskSpace>(*current_task);
            active_task_spaces.enqueue(task_space);
            current_task->children = task_space;
        }

        return current_task->children;
    }
    else
        return top_space;
}

unsigned scope_depth()
{
    if( auto ts = current_task_space() )
        return ts->depth;
    else
        return 0;
}

/*! Create an event on which the termination of the current task depends.
 *  A task must currently be running.
 *
 * @return Handle to flag the event with `reach_event` later.
 *         nullopt if there is no task running currently
 */
std::optional< scheduler::EventPtr > create_event()
{
    if( current_task )
        return current_task->make_event();
    else
        return std::nullopt;
}

//! get backtrace from currently running task
std::vector<std::reference_wrapper<Task>> backtrace()
{
    std::vector<std::reference_wrapper<Task>> bt;
    Task * task = current_task;

    while( task )
    {
        bt.push_back(*task);
        task = task->space->parent;
    }

    return bt;
}

void init( std::shared_ptr<scheduler::IScheduler> scheduler )
{
    top_space = std::make_shared<TaskSpace>();
    active_task_spaces.enqueue(top_space);
    top_scheduler = scheduler;
}

void init( size_t n_threads )
{
    init(std::make_shared<scheduler::DefaultScheduler>(n_threads));
}

/*! wait until all tasks in the current task space finished
 */
void barrier()
{
    while( ! top_space->empty() )
        idle();
}

void finalize()
{
    barrier();
    top_scheduler->stop();
    top_scheduler.reset();
    top_space.reset();
}

//! pause the currently running task at least until event is reached
void yield( scheduler::EventPtr event )
{
    while( ! event->is_reached() )
    {
        if( current_task )
            current_task->yield(event);
        else
	{
            {
                std::lock_guard<std::mutex> lock( event->waker_mutex );
                event->waker = top_scheduler;
            }

            if( ! event->is_reached() )
                idle();
	}
    }
}
/*
void update_active_task_spaces()
{
    SPDLOG_TRACE("update active task spaces");
    std::vector< std::shared_ptr< TaskSpace > > buf;
    std::shared_ptr< TaskSpace > space;

    static std::mutex m;
    std::lock_guard<std::mutex> l(m);

    while(active_task_spaces.try_dequeue(space))
    {
        SPDLOG_TRACE("update task space {}", (void*)space.get());
        space->init_until_ready();

        bool remove_space = false;
        if( auto parent = space->parent )
        {
            if(
                space->empty() &&
                parent->is_dead()
            )
            {
                parent->space->try_remove( *parent );
                remove_space = true;
            }
        }

        if(! remove_space)
            buf.push_back(space);
    }

    for( auto space : buf )
        active_task_spaces.enqueue(space);
}
*/
void schedule()
{
    auto ts = top_scheduler;
    if(ts)
    {
        //update_active_task_spaces();
        auto space = top_space;
        if(space)
            space->init_until_ready();

        ts->schedule();
    }
}

void schedule( dispatch::thread::WorkerThread & worker )
{
    auto ts = top_scheduler;
    if(ts)
    {
        //update_active_task_spaces();
        auto space = top_space;
        if(space)
            space->init_until_ready();

        ts->schedule( worker );
    }    
}


//! apply a patch to the properties of the currently running task
void update_properties(typename TaskProperties::Patch const& patch)
{
    if( current_task )
    {
        current_task->apply_patch(patch);
        current_task->update_graph();
    }
    else
        throw std::runtime_error("update_properties: currently no task running");
}

} // namespace redGrapes

