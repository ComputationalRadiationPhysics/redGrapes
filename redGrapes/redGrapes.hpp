/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once


#include <redGrapes/scheduler/default_scheduler.hpp>
#include <redGrapes/context.hpp>
#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/future.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/task/task_builder.hpp>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/util/trace.hpp>
#include <redGrapes/memory/allocator.hpp>
#include <redGrapes/dispatch/dispatcher.hpp>
#include <redGrapes/scheduler/scheduler.hpp>

#include <spdlog/spdlog.h>
#include <type_traits>

namespace redGrapes
{

void init_tracing();
void cpubind_mainthread();

void init( size_t n_workers, std::shared_ptr<scheduler::IScheduler> scheduler);
void init( size_t n_workers = std::thread::hardware_concurrency() );
void finalize();

//! wait until all tasks in the current task space finished
void barrier();

//! pause the currently running task at least until event is reached
void yield(scheduler::EventPtr event);

//! apply a patch to the properties of the currently running task
void update_properties(typename TaskProperties::Patch const& patch);

//! get backtrace from currently running task
std::vector<std::reference_wrapper<Task>> backtrace();

/*! Create an event on which the termination of the current task depends.
 *  A task must currently be running.
 *
 * @return Handle to flag the event with `reach_event` later.
 *         nullopt if there is no task running currently
 */
std::optional<scheduler::EventPtr> create_event();

/*! create a new task, as child of the currently running task (if there is one)
 *
 * @param f callable that takes "proprty-building" objects as args
 * @param args are forwarded to f after the each arg added its
 *             properties to the task
 *
 * For the argument-types can a trait be implemented which
 * defines a hook to add task properties depending the the
 * argument.
 *
 * @return future from f's result
 */
template<typename Callable, typename... Args>
auto emplace_task(Callable&& f, Args&&... args)
{
    dispatch::thread::WorkerId worker_id =
     // linear
	    next_worker % worker_pool->size();

     // interleaved
//    2*next_worker % worker_pool->size() + ((2*next_worker) / worker_pool->size())%2;

    next_worker++;
    memory::current_arena = worker_id;

    SPDLOG_TRACE("emplace task to worker {} next_worker={}", worker_id, next_worker);

    return std::move(TaskBuilder< Callable, Args... >( std::move(f), std::forward<Args>(args)... ));
}

} // namespace redGrapes
