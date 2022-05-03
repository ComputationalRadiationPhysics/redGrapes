/* Copyright 2019-2021 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <future>
#include <moodycamel/concurrentqueue.h>
#include <redGrapes/context.hpp>
#include <redGrapes/dispatch/dispatcher.hpp>
#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/delayed_functor.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/task/task_result.hpp>
#include <shared_mutex>
#include <spdlog/spdlog.h>
#include <sstream>
#include <type_traits>
#include <unordered_map>

namespace redGrapes
{

    std::shared_ptr<TaskSpace> current_task_space();

    /*! Create an event on which the termination of the current task depends.
     *  A task must currently be running.
     *
     * @return Handle to flag the event with `reach_event` later.
     *         nullopt if there is no task running currently
     */
    std::optional<scheduler::EventPtr> create_event();

    //! get backtrace from currently running task
    std::vector<std::shared_ptr<Task>> backtrace();

    void init_default(size_t n_threads = std::thread::hardware_concurrency());
    void finalize();

    //! pause the currently running task at least until event is reached
    void yield(scheduler::EventPtr event);

    /*! wait until all tasks in the current task space finished
     */
    void barrier();

    void remove_task(std::shared_ptr<Task> task);

    void update_active_task_spaces();

    //! apply a patch to the properties of the currently running task
    void update_properties(typename TaskProperties::Patch const& patch);


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
    TaskResult<typename std::result_of<Callable(Args...)>::type> emplace_task(Callable&& f, Args&&... args)
    {
        typename TaskProperties::Builder builder;
        return emplace_task(f, std::move(builder), std::forward<Args>(args)...);
    }


    template<typename... Args>
    static inline void pass(Args&&...)
    {
    }

    struct PropBuildHelper
    {
        typename TaskProperties::Builder& builder;

        template<typename T>
        inline int build(T const& x)
        {
            trait::BuildProperties<T>::build(builder, x);
            return 0;
        }

        void foo()
        {
        }
    };

    /*! create a new task, as child of the currently running task (if there is one)
     *
     * @param f callable that takes "proprty-building" objects as args
     * @param builder used sequentially by property-builders of each arg
     * @param args are forwarded to f after the each arg added its
     *             properties to the task
     *
     * Firstly the task properties get initialized through
     * the builder-object.
     * Secondly, for the argument-types can a trait be implemented which
     * defines a hook to add further task properties depending the the
     * argument.
     *
     * @return future from f's result
     */
    template<typename Callable, typename... Args>
    TaskResult<typename std::result_of<Callable(Args...)>::type> emplace_task(
        Callable&& f,
        typename TaskProperties::Builder builder,
        Args&&... args)
    {
        PropBuildHelper build_helper{builder};
        pass(build_helper.template build<Args>(args)...);

        build_helper.foo();

        auto impl = std::bind(f, std::forward<Args>(args)...);

        /* todo: std::packaged_task could replace the internal DelayedFunctor,
         * but how exactly ?
         */
        // std::packaged_task< typename std::result_of< Callable() >::type() > delayed(std::move(impl));

        auto delayed = make_delayed_functor(std::move(impl));
        auto future = delayed.get_future();

        builder.init_id(); // needed because property builder may be copied from a template

        auto task = make_fun_task(
            std::bind(
                [](auto&& delayed, Task& task) mutable
                {
                    delayed();
                    task.get_result_event().notify();
                },
                std::move(delayed),
                std::placeholders::_1),
            (TaskProperties&&) builder);

        SPDLOG_DEBUG("RedGrapes::emplace_task {}\n", (TaskProperties const&) *task);

        task->xty_task = task;
        task->scope_depth = scope_depth() + 1;

        current_task_space()->push(task);
        top_scheduler->notify();

        return make_task_result(std::move(future), task->get_result_event());
    }

} // namespace redGrapes
