/* Copyright 2019-2021 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/scheduler/default_scheduler.hpp>

namespace redGrapes
{

thread_local std::shared_ptr<Task> current_task;
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
            auto task_space = std::make_shared<TaskSpace>(
                std::make_shared<PrecedenceGraph<ResourceUser>>(),
                current_task);

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
    if( current_task )
        return current_task->scope_depth;
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
std::vector<std::shared_ptr<Task>> backtrace()
{
    std::vector<std::shared_ptr<Task>> bt;
    std::shared_ptr<Task> task = current_task;

    while( task )
    {
        bt.push_back(task);
        task = task->space->parent.lock();
    }

    return bt;
}

void init_default( size_t n_threads )
{
    top_space = std::make_shared<TaskSpace>(std::make_shared<PrecedenceGraph<ResourceUser>>());
    active_task_spaces.enqueue(top_space);
    top_scheduler = std::make_shared<scheduler::DefaultScheduler>(n_threads);
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

    top_scheduler = std::shared_ptr<scheduler::IScheduler>();
    top_space = std::shared_ptr<TaskSpace>();
}

//! pause the currently running task at least until event is reached
void yield( scheduler::EventPtr event )
{
    while(! event->is_reached() )
    {
        if( current_task )
            current_task->yield(event);
        else
            idle();
    }
}

void remove_task(std::shared_ptr<Task> task)
{
    SPDLOG_TRACE("remove task {}", task->task_id);
    if( auto task_space = task->space )
    {
        task_space->remove(task);
        top_scheduler->notify();
    }
}

void update_active_task_spaces()
{
    SPDLOG_TRACE("update active task spaces");
    std::vector< std::shared_ptr< TaskSpace > > buf;

    std::shared_ptr< TaskSpace > space;
    while(active_task_spaces.try_dequeue(space))
    {
        while(auto new_task = space->next())
        {
            new_task->sg_init();
            new_task->pre_event.up();
            new_task->get_pre_event().notify();
        }

        bool remove = false;
        if( auto parent = space->parent.lock() )
        {
            if(
               space->empty()
               && parent->is_finished()
               )
            {
                remove_task(parent);
                remove = true;
            }
        }

        if(! remove)
            buf.push_back(space);
    }

    for( auto space : buf )
        active_task_spaces.enqueue(space);
}

//! apply a patch to the properties of the currently running task
void update_properties(typename TaskProperties::Patch const& patch)
{
    if( current_task )
    {
        current_task->apply_patch(patch);
        current_task_space()->precedence_graph->update_dependencies(current_task);
    }
    else
        throw std::runtime_error("update_properties: currently no task running");
}

} // namespace redGrapes

