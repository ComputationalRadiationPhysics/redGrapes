/* Copyright 2019-2021 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <unordered_set>
#include <optional>
#include <atomic>

#include <moodycamel/concurrentqueue.h>

#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/thread_local.hpp>
#include <redGrapes/imanager.hpp>

namespace redGrapes
{
namespace scheduler
{

struct FIFOSchedulerProp
{
    std::atomic_flag in_ready_list = ATOMIC_FLAG_INIT;
    std::atomic_flag in_running_list = ATOMIC_FLAG_INIT;

    FIFOSchedulerProp()
    {
    }
    FIFOSchedulerProp(FIFOSchedulerProp&& other)
    {
    }
    FIFOSchedulerProp(FIFOSchedulerProp const& other)
    {
    }
    FIFOSchedulerProp& operator=(FIFOSchedulerProp const& other)
    {
        return *this;
    }

    template<typename PropertiesBuilder>
    struct Builder
    {
        PropertiesBuilder& builder;

        Builder(PropertiesBuilder& b) : builder(b)
        {
        }
    };
};

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
            //spdlog::info("FIFO: run task {}", task_id);

            mgr.get_scheduling_graph()->task_start( task_id );

            mgr.current_task() = task_vertex;
            //spdlog::info("call task impl");
            bool finished = (*(*task_vertex)->task->impl)();
            mgr.current_task() = std::nullopt;

            if(finished)
            {
                //spdlog::info("FIFO: finished task {}", task_id);
                if(auto children = (*task_vertex)->children)
                    while(auto new_task = (*children)->next())
                        mgr.activate_task(*new_task);

                mgr.get_scheduling_graph()->task_end(task_id);

                if(! (*task_vertex)->children)
                    mgr.remove_task(*task_vertex);
            }
            /*
            else
                spdlog::info("FIFO: paused task {}", task_id);
            */

            return true;
        }
        else
            return false;
    }

    // precedence graph must be locked
    bool activate_task( TaskVertexPtr task_vertex )
    {
        auto task_id = task_vertex->task->task_id;
        if(mgr.get_scheduling_graph()->is_task_ready(task_id))
        {
            if(!task_vertex->task->in_ready_list.test_and_set())
            {
                //spdlog::info("FIFO: task {} is ready", task_id);
                ready.enqueue(task_vertex);
                mgr.get_scheduler()->notify();

                return true;
            }
        }

        return false;
    }

private:
    std::optional<TaskVertexPtr> get_job()
    {
        if( auto task_vertex = try_next_task() )
            return task_vertex;
        else
        {
            mgr.update_active_task_spaces();
            return try_next_task();
        }
    }

    std::optional<TaskVertexPtr> try_next_task()
    {
        do
        {
            TaskVertexPtr task_vertex;
            if(ready.try_dequeue(task_vertex))
                return task_vertex;
        }
        while( mgr.activate_next() );

        return std::nullopt;
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


template<>
struct fmt::formatter<redGrapes::scheduler::FIFOSchedulerProp>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::scheduler::FIFOSchedulerProp const& prop, FormatContext& ctx)
    {
        auto out = ctx.out();
        format_to(out, "\"active\": 0");
        return out;
    }
};


