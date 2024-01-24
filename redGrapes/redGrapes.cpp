/* Copyright 2019-2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/dispatch/thread/worker_pool.hpp>
#include <redGrapes/memory/allocator.hpp>
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/scheduler/default_scheduler.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/util/trace.hpp>

#include <moodycamel/concurrentqueue.h>

#include <functional>
#include <memory>
#include <optional>

#if REDGRAPES_ENABLE_TRACE
PERFETTO_TRACK_EVENT_STATIC_STORAGE();
#endif

namespace redGrapes
{

    thread_local Task* Context::current_task;
    thread_local std::function<void()> Context::idle;
    thread_local unsigned Context::next_worker;
    thread_local unsigned Context::current_arena;
    thread_local scheduler::WakerId Context::current_waker_id;
    thread_local std::shared_ptr<dispatch::thread::WorkerThread> Context::current_worker;

    Context::Context()
    {
        idle = [this] { this->scheduler->idle(); };
    }

    Context::~Context()
    {
    }

    std::shared_ptr<TaskSpace> Context::current_task_space() const
    {
        if(current_task)
        {
            if(!current_task->children)
            {
                auto task_space = std::make_shared<TaskSpace>(current_task);
                SPDLOG_TRACE("create child space = {}", (void*) task_space.get());
                current_task->children = task_space;

                std::unique_lock<std::shared_mutex> wr_lock(current_task->space->active_child_spaces_mutex);
                current_task->space->active_child_spaces.push_back(task_space);
            }

            return current_task->children;
        }
        else
            return root_space;
    }

    unsigned Context::scope_depth() const
    {
        if(auto ts = current_task_space())
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
    std::optional<scheduler::EventPtr> Context::create_event()
    {
        if(current_task)
            return current_task->make_event();
        else
            return std::nullopt;
    }

    //! get backtrace from currently running task
    std::vector<std::reference_wrapper<Task>> Context::backtrace()
    {
        std::vector<std::reference_wrapper<Task>> bt;
        for(Task* task = current_task; task != nullptr; task = task->space->parent)
            bt.push_back(*task);

        return bt;
    }

    void Context::init_tracing()
    {
#if REDGRAPES_ENABLE_TRACE
        perfetto::TracingInitArgs args;
        args.backends |= perfetto::kInProcessBackend;
        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();

        tracing_session = StartTracing();
#endif
    }

    void Context::finalize_tracing()
    {
#if REDGRAPES_ENABLE_TRACE
        StopTracing(tracing_session);
#endif
    }

    void Context::init(size_t n_workers, std::shared_ptr<scheduler::IScheduler> scheduler)
    {
        init_tracing();

        this->n_workers = n_workers;
        worker_pool = std::make_shared<dispatch::thread::WorkerPool>(hwloc_ctx, n_workers);
        worker_pool->emplace_workers(n_workers);

        root_space = std::make_shared<TaskSpace>();
        this->scheduler = scheduler;

        worker_pool->start();
    }

    void Context::init(size_t n_workers)
    {
        init(n_workers, std::make_shared<scheduler::DefaultScheduler>());
    }

    /*! wait until all tasks in the current task space finished
     */
    void Context::barrier()
    {
        SPDLOG_TRACE("barrier");

        while(!root_space->empty())
            idle();
    }

    void Context::finalize()
    {
        barrier();

        worker_pool->stop();

        scheduler.reset();
        root_space.reset();

        finalize_tracing();
    }

    //! pause the currently running task at least until event is reached
    void Context::yield(scheduler::EventPtr event)
    {
        if(current_task)
        {
            while(!event->is_reached())
                current_task->yield(event);
        }
        else
        {
            event->waker_id = Context::current_waker_id;
            while(!event->is_reached())
                idle();
        }
    }

    //! apply a patch to the properties of the currently running task
    void Context::update_properties(typename TaskProperties::Patch const& patch)
    {
        if(current_task)
        {
            current_task->apply_patch(patch);
            current_task->update_graph();
        }
        else
            throw std::runtime_error("update_properties: currently no task running");
    }

} // namespace redGrapes
