/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <redGrapes/context.hpp>
#include <redGrapes/dispatch/dispatcher.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/future.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/task/task_space.hpp>
#include <spdlog/spdlog.h>
#include <type_traits>

namespace redGrapes
{

    /* HELPERS */

    std::shared_ptr<TaskSpace> current_task_space();
    void update_active_task_spaces();

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


    /* USER INTERFACE */
    void init(std::shared_ptr<scheduler::IScheduler> scheduler);
    void init(size_t n_threads = std::thread::hardware_concurrency());

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
    Future<typename std::result_of<Callable(Args...)>::type> emplace_task(Callable&& f, Args&&... args)
    {
        typename TaskProperties::Builder builder;
        return emplace_task(f, std::move(builder), std::forward<Args>(args)...);
    }

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
    Future<typename std::result_of<Callable(Args...)>::type> emplace_task(
        Callable&& f,
        typename TaskProperties::Builder builder,
        Args&&... args)
    {
        PropBuildHelper build_helper{builder};
        pass(build_helper.template build<Args>(args)...);

        build_helper.foo();

        auto impl = std::bind(f, std::forward<Args>(args)...);
        /*
        auto impl = [f=std::move(f), args...]() mutable {
            return f(std::forward<Args>(args)...);
        };
*/
        builder.init_id();

        SPDLOG_DEBUG("redGrapes::emplace_task {}", (TaskProperties const&)builder);
        Task& task = current_task_space()->emplace_task(std::move(impl), (TaskProperties &&) builder);

        return Future<typename std::result_of<Callable(Args...)>::type>(task);
    }

} // namespace redGrapes
