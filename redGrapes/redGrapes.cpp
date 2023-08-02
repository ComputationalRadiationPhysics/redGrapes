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
#include <redGrapes/util/multi_arena_alloc.hpp>
#include <redGrapes/scheduler/default_scheduler.hpp>

#include <redGrapes/util/trace.hpp>

#if REDGRAPES_ENABLE_TRACE
PERFETTO_TRACK_EVENT_STATIC_STORAGE();
#endif

namespace redGrapes
{

thread_local Task * current_task;
thread_local std::function<void()> idle;

namespace memory
{
std::shared_ptr< MultiArenaAlloc > alloc;
thread_local unsigned current_arena;
} // namespace memory

std::shared_ptr< TaskSpace > top_space;
std::shared_ptr< scheduler::IScheduler > top_scheduler;

#if REDGRAPES_ENABLE_TRACE
std::shared_ptr< perfetto::TracingSession > tracing_session;
#endif

std::shared_ptr<TaskSpace> current_task_space()
{
    if( current_task )
    {
        if( ! current_task->children )
        {
            auto task_space = std::make_shared<TaskSpace>(current_task);
            SPDLOG_TRACE("create child space = {}", (void*)task_space.get());
            current_task->children = task_space;

            std::unique_lock< std::shared_mutex > wr_lock( current_task->space->active_child_spaces_mutex );
            current_task->space->active_child_spaces.push_back( task_space );
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
    for(
        Task * task = current_task;
        task != nullptr;
        task = task->space->parent
    )
        bt.push_back(*task);

    return bt;
}

void init( size_t n_threads )
{
    // use one arena with 8 MiB chunksize per worker
    memory::alloc = std::make_shared< memory::MultiArenaAlloc >( 32 * 1024 * 1024, n_threads );

#if REDGRAPES_ENABLE_TRACE
    perfetto::TracingInitArgs args;
    args.backends |= perfetto::kInProcessBackend;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

    tracing_session = StartTracing();
#endif

    top_space = std::make_shared<TaskSpace>();
    top_scheduler = std::make_shared<scheduler::DefaultScheduler>(n_threads);
    top_scheduler->start();
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

#if REDGRAPES_ENABLE_TRACE
    StopTracing( tracing_session );
#endif
}

//! pause the currently running task at least until event is reached
void yield( scheduler::EventPtr event )
{
    if( current_task )
    {
        while( ! event->is_reached() )
            current_task->yield(event);
    }
    else
    {
        event->waker_id = dispatch::thread::current_waker_id;
        while( ! event->is_reached() )
            idle();
    }
}

Task * schedule( dispatch::thread::WorkerThread & worker )
{
    auto sched = top_scheduler;
    auto space = top_space;

    if( sched && space )
        return sched->schedule(worker);

    return nullptr;
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

