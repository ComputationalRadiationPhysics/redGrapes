/* Copyright 2022-2023 The RedGrapes Community.
 *
 * Authors: Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <memory> // std::shared_ptr
#include <spdlog/spdlog.h>

#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/scheduler/scheduler.hpp>

//#include <redGrapes/task/future.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/memory/hwloc_alloc.hpp>
#include <redGrapes/dispatch/thread/worker.hpp>

namespace redGrapes
{

struct Context
{
    Context();
    ~Context();

    void init_tracing();
    void finalize_tracing();

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

    unsigned scope_depth() const;
    std::shared_ptr<TaskSpace> current_task_space() const;

    void execute_task( Task & task );

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
    template< typename Callable, typename... Args >
    auto emplace_task(Callable&& f, Args&&... args);

    static thread_local Task * current_task;
    static thread_local std::function< void () > idle;
    static thread_local unsigned next_worker;
    
    static thread_local scheduler::WakerId current_waker_id;
    static thread_local std::shared_ptr< dispatch::thread::WorkerThread > current_worker;

    unsigned current_arena;
    HwlocContext hwloc_ctx;
    std::shared_ptr< dispatch::thread::WorkerPool > worker_pool;

    std::shared_ptr< TaskSpace > root_space;
    std::shared_ptr< scheduler::IScheduler > scheduler;

#if REDGRAPES_ENABLE_TRACE
    std::shared_ptr< perfetto::TracingSession > tracing_session;
#endif
};


/* ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
 *  S I N G L E T O N
 * ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
 */

struct SingletonContext
{
    inline static Context & get()
    {
        static Context ctx;
        return ctx;
    }
};

inline void init( size_t n_workers, std::shared_ptr<scheduler::IScheduler> scheduler ) {
    SingletonContext::get().init( n_workers, scheduler); }

inline void init( size_t n_workers = std::thread::hardware_concurrency() ) {
    SingletonContext::get().init( n_workers ); }

inline void finalize() {
    SingletonContext::get().finalize(); }

inline void barrier() {
    SingletonContext::get().barrier(); }

inline void yield(scheduler::EventPtr event) {
    SingletonContext::get().yield(event); }

inline void update_properties(typename TaskProperties::Patch const& patch) {
    SingletonContext::get().update_properties( patch ); }

inline std::vector<std::reference_wrapper<Task>> backtrace() {
    return SingletonContext::get().backtrace(); }

inline std::optional<scheduler::EventPtr> create_event() {
    return SingletonContext::get().create_event(); }

inline unsigned scope_depth() {
    return SingletonContext::get().scope_depth(); }

inline std::shared_ptr<TaskSpace> current_task_space() {
    return SingletonContext::get().current_task_space(); }

template<typename Callable, typename... Args>
inline auto emplace_task(Callable&& f, Args&&... args) {
    return std::move(
               SingletonContext::get().emplace_task(
                   std::move(f),
                   std::forward<Args>(args)...
               )
           ); }

} //namespace redGrapes



// `TaskBuilder` needs "Context`, so can only include here after definiton
#include <redGrapes/task/task_builder.hpp>

namespace redGrapes
{
    template<typename Callable, typename... Args>
    auto Context::emplace_task(Callable&& f, Args&&... args)
    {
        dispatch::thread::WorkerId worker_id =
         // linear
    	    next_worker % worker_pool->size();

         // interleaved
    //    2*next_worker % worker_pool->size() + ((2*next_worker) / worker_pool->size())%2;

        next_worker++;
        current_arena = worker_id;

        SPDLOG_TRACE("emplace task to worker {} next_worker={}", worker_id, next_worker);

        return std::move(TaskBuilder< Callable, Args... >( std::move(f), std::forward<Args>(args)... ));
    }
} // namespace redGrapes

