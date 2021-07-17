/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <queue>
#include <optional>

#include <moodycamel/concurrentqueue.h>

#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/thread_local.hpp>
#include <redGrapes/imanager.hpp>

namespace redGrapes
{
namespace scheduler
{

template < typename Task >
struct FIFO : public IScheduler< Task >
{
    using TaskVertexPtr = std::shared_ptr<PrecedenceGraphVertex<Task>>;

    IManager<Task>& mgr;

    moodycamel::ConcurrentQueue< TaskVertexPtr > ready;
    moodycamel::ConcurrentQueue< TaskVertexPtr > running;

    FIFO(IManager<Task>& mgr) : mgr(mgr)
    {
    }

    //! returns true if a job was consumed, false if queue is empty
    bool consume()
    {
        if( auto task_vertex = get_job() )
        {
            auto task_id = (*task_vertex)->task->task_id;

            mgr.get_scheduling_graph()->task_start( task_id );

            running.enqueue( *task_vertex );

            bool finished = this->mgr.run_task( *task_vertex );

            if( finished )
                mgr.get_scheduling_graph()->task_end( task_id );
            
            return true;
        }
        else
            return false;
    }

    // precedence graph must be locked
    void activate_task( TaskVertexPtr task_vertex )
    {
        auto task_id = task_vertex->task->task_id;

        if( ! mgr.get_scheduling_graph()->exists_task( task_id ) )
            mgr.get_scheduling_graph()->add_task( task_vertex );

        if( ! mgr.get_scheduling_graph()->is_task_finished( task_id ) )
            if( mgr.get_scheduling_graph()->is_task_ready( task_id ) )
                ready.enqueue( task_vertex );
    }

private:
    std::optional<TaskVertexPtr> get_job()
    {
        TaskVertexPtr task_vertex;
        spdlog::trace("FIFO::get_job()");

        if(ready.try_dequeue(task_vertex))
            return task_vertex;

        update_running_spaces();

        if(ready.try_dequeue(task_vertex))
            return task_vertex;

        update_main_space();

        if(ready.try_dequeue(task_vertex))
            return task_vertex;

        spdlog::trace("FIFO::get_job(): no job available");
        
        return std::nullopt;
    }

    void update_running_spaces()
    {
        spdlog::trace("FIFO::update_running_spaces()");
        size_t len = running.size_approx();
        for(size_t i = 0; i < len; ++i)
        {
            TaskVertexPtr task_vertex;
            if(running.try_dequeue(task_vertex))
            {
                TaskID task_id = task_vertex->task->task_id;

                if(auto children = task_vertex->children)
                    if(auto new_task = (*children)->next())
                        mgr.activate_task(*new_task);

                if(!mgr.get_scheduling_graph()->is_task_finished(task_id))
                    running.enqueue(task_vertex);
                else
                {
                    // activate followers
                    std::shared_lock<std::shared_mutex> rdlock(task_vertex->out_edges_mutex);
                    for(auto following_task : task_vertex->out_edges)
                        mgr.activate_task(following_task.lock());

                    mgr.notify();
                }
            }
        }
    }

    void update_main_space()
    {
        spdlog::trace("FIFO::update_main_space()");
        if(auto task_vertex = mgr.current_task_space()->next())
            mgr.activate_task(*task_vertex);
    }
};

/*! Factory function to easily create a fifo-scheduler object
 */
template <
    typename Task
>
auto make_fifo_scheduler(
    IManager< Task > & m
)
{
    return std::make_shared<
               FIFO< Task >
           >(m);
}

} // namespace scheduler

} // namespace redGrapes

