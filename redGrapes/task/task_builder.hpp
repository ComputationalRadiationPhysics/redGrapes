/* Copyright 2023-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "redGrapes/task/future.hpp"
#include "redGrapes/task/task.hpp"
#include "redGrapes/task/task_space.hpp"
#include "redGrapes/util/bind_args.hpp"

#include <spdlog/spdlog.h>

#include <type_traits>
#include <utility>

namespace redGrapes
{

    /* HELPERS */

    template<typename... Args>
    static inline void pass(Args&&...)
    {
    }

    template<typename TTask, typename B>
    struct PropBuildHelper
    {
        typename TTask::TaskProperties::template Builder<B>& builder;

        template<typename T>
        inline int build(T const& x)
        {
            trait::BuildProperties<T, TTask>::build(builder, x);
            return 0;
        }
    };

    // Helper function to conditionally forward the argument
    template<typename T>
    constexpr auto forward_arg(T&& arg)
    {
        if constexpr(traits::is_derived_from_pair_v<std::decay_t<T>>)
        {
            if constexpr(std::is_same_v<typename std::decay_t<T>::second_type, ResourceAccess>)
            {
                return std::forward<typename T::first_type>(arg.first);
            }
            else
            {
                return std::forward<T>(arg);
            }
        }
        else
        {
            return std::forward<T>(arg);
        }
    }

    /* TASK BUILDER */

    template<typename TTask, typename Callable, typename... Args>
    struct TaskBuilder : TTask::TaskProperties::template Builder<TaskBuilder<TTask, Callable, Args...>>
    {
        using Impl = typename std::invoke_result_t<
            BindArgs<Callable, decltype(forward_arg(std::declval<Args>()))...>,
            Callable,
            decltype(forward_arg(std::declval<Args>()))...>;
        using Result = typename std::invoke_result_t<Callable, decltype(forward_arg(std::declval<Args>()))...>;

        std::shared_ptr<TaskSpace> space;
        FunTask<Impl, TTask>* task;

        TaskBuilder(
            FunTask<Impl, TTask>* task,
            std::shared_ptr<TaskSpace> const& current_space,
            Callable&& f,
            Args&&... args)
            : TTask::TaskProperties::template Builder<TaskBuilder>(*this)
            , space(current_space)
            , task{task}
        {
            // init properties from args
            PropBuildHelper<TTask, TaskBuilder> build_helper{*this};
            pass(build_helper.template build<Args>(std::forward<Args>(args))...);

            // init id
            this->init_id();

            // set impl
            task->impl.emplace(BindArgs<Callable, decltype(forward_arg(std::declval<Args>()))...>{}(
                std::move(f),
                forward_arg(std::forward<Args>(args))...));
        }

        TaskBuilder(TaskBuilder& other)
            : TTask::TaskProperties::template Builder<TaskBuilder>(*this)
            , space(other.space)
            , task(other.task)
        {
            other.task = nullptr;
        }

        TaskBuilder(TaskBuilder&& other)
            : TTask::TaskProperties::template Builder<TaskBuilder>(*this)
            , space(std::move(other.space))
            , task(std::move(other.task))
        {
            other.task = nullptr;
        }

        ~TaskBuilder()
        {
            if(task)
                submit();
        }

        TaskBuilder& enable_stack_switching()
        {
            task->enable_stack_switching = true;
            return *this;
        }

        auto submit()
        {
            TTask* t = task;
            task = nullptr;

            SPDLOG_TRACE("submit task {}", (typename TTask::TaskProperties const&) *t);
            space->submit(t);
            t->scheduler_p->emplace_task(*t);

            return Future<Result, TTask>(*t);
        }

        auto get()
        {
            return submit().get();
        }
    };

} // namespace redGrapes
