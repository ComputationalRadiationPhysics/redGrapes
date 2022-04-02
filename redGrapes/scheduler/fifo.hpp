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
#include <redGrapes/dispatch/thread/worker.hpp>
#include <redGrapes/dispatch/thread/thread_local.hpp>
#include <redGrapes/imanager.hpp>

namespace redGrapes
{
namespace scheduler
{

struct FIFOSchedulerProp
{
    std::atomic_flag in_activation_queue = ATOMIC_FLAG_INIT;
    std::atomic_flag in_ready_list = ATOMIC_FLAG_INIT;

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

    struct Patch
    {
        template <typename PatchBuilder>
        struct Builder
        {
            Builder( PatchBuilder & ) {}
        };
    };

    void apply_patch( Patch const & ) {};
};

template < typename Task >
struct FIFO : public IScheduler< Task >
{
    IManager<Task>& mgr;

    moodycamel::ConcurrentQueue< TaskVertexPtr > ready;
    moodycamel::ConcurrentQueue< TaskVertexPtr > running;

    FIFO(IManager<Task>& mgr) : mgr(mgr)
    {
    }

    void activate_task( TaskVertexPtr task_vertex )
    {
        ready.enqueue(task_vertex);        
    }

    /*! take a job from the ready queue
     * if none available, update 
     */
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

    /*! call the manager to activate tasks until we get at least
     * one in the ready queue
     */
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


