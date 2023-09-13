/* Copyright 2019-2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <optional>
#include <functional>
#include <memory>
#include <moodycamel/concurrentqueue.h>

#include <redGrapes/util/multi_arena_alloc.hpp>
#include <redGrapes/dispatch/thread/worker_pool.hpp>
#include <redGrapes/scheduler/scheduler.hpp>

#include <redGrapes/redGrapes.hpp>

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
std::shared_ptr< dispatch::thread::WorkerPool > worker_pool;
std::shared_ptr< scheduler::IScheduler > top_scheduler;

#if REDGRAPES_ENABLE_TRACE
std::shared_ptr< perfetto::TracingSession > tracing_session;
#endif

hwloc_topology_t topology;

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

void init_allocator( size_t n_arenas, size_t chunk_size )
{
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    memory::alloc = std::make_shared< memory::MultiArenaAlloc >( chunk_size - sizeof(memory::BumpAllocChunk), n_arenas );
}

void init_tracing()
{
#if REDGRAPES_ENABLE_TRACE
    perfetto::TracingInitArgs args;
    args.backends |= perfetto::kInProcessBackend;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

    tracing_session = StartTracing();
#endif    
}

void cpubind_mainthread()
{
    size_t n_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU );
    hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, 1 );

    if( hwloc_set_cpubind(topology, obj->cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT) )    {
        char *str;
        int error = errno;
        hwloc_bitmap_asprintf(&str, obj->cpuset);
        spdlog::warn("Couldn't cpubind to cpuset {}: {}\n", str, strerror(error));
        free(str);
    }    
}

void init( size_t n_workers, std::shared_ptr<scheduler::IScheduler> scheduler)
{
    init_tracing();

    top_space = std::make_shared<TaskSpace>();
    worker_pool = std::make_shared<dispatch::thread::WorkerPool>( n_workers );
    top_scheduler = scheduler;

    worker_pool->start();

    /* bind main thread to
     */
    cpubind_mainthread();
}

void init( size_t n_workers )
{
    init_allocator( n_workers );
    init( n_workers, std::make_shared<scheduler::DefaultScheduler>());
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
    worker_pool->stop();

    top_scheduler.reset();
    top_space.reset();

    hwloc_topology_destroy(topology);
    
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

