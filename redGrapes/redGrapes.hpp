/* Copyright 2022-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/SchedulerDescription.hpp"
#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/globalSpace.hpp"
#include "redGrapes/scheduler/event.hpp"
#include "redGrapes/scheduler/pool_scheduler.hpp"
#include "redGrapes/task/task.hpp"
#include "redGrapes/task/task_builder.hpp"
#include "redGrapes/task/task_space.hpp"
#include "redGrapes/util/bind_args.hpp"
#include "redGrapes/util/tuple_map.hpp"

#include <boost/mp11.hpp>
#include <spdlog/spdlog.h>

#include <memory>
#include <new>

namespace redGrapes
{

    template<typename TSchedMap, C_TaskProperty... TUserTaskProperties>
    struct RedGrapes
    {
    public:
        using RGTask = Task<TUserTaskProperties...>;

        template<typename... TSchedulerDesc>
        RedGrapes(TSchedulerDesc... execDescs)
        {
            init_tracing();

            (..., (scheduler_map[(typename TSchedulerDesc::Key{})] = execDescs.scheduler));

            // TODO find n_workers without making a tuple
            auto execDescTuple = std::make_tuple(execDescs...);
            TaskFreeCtx::n_workers
                = std::apply([](auto... args) { return (args.scheduler->n_workers + ...); }, execDescTuple);

            if(TaskFreeCtx::n_workers > TaskFreeCtx::n_pus)
                SPDLOG_WARN(
                    "{} worker-threads requested, but only {} PUs available!",
                    TaskFreeCtx::n_workers,
                    TaskFreeCtx::n_pus);

            TaskFreeCtx::worker_alloc_pool.allocs.reserve(TaskFreeCtx::n_workers);

            root_space = std::make_shared<TaskSpace>();

            auto initAdd = [](auto scheduler, auto& base_worker_id)
            {
                scheduler->init(base_worker_id);
                base_worker_id = base_worker_id + scheduler->n_workers;
            };
            WorkerId base_worker_id = 0;
            std::apply(
                [&base_worker_id, initAdd](auto... args) { ((initAdd(args.scheduler, base_worker_id)), ...); },
                execDescTuple);

            boost::mp11::mp_for_each<TSchedMap>(
                [&](auto pair) { scheduler_map[boost::mp11::mp_first<decltype(pair)>{}]->startExecution(); });
        }

        ~RedGrapes()
        {
            barrier();

            boost::mp11::mp_for_each<TSchedMap>(
                [&](auto pair) { scheduler_map[boost::mp11::mp_first<decltype(pair)>{}]->stopExecution(); });
            boost::mp11::mp_for_each<TSchedMap>([&](auto pair)
                                                { scheduler_map[boost::mp11::mp_first<decltype(pair)>{}].reset(); });
            root_space.reset();

            finalize_tracing();
        }

        void init_tracing();
        void finalize_tracing();

        //! wait until all tasks in the current task space finished
        void barrier();

        //! pause the currently running task at least until event is reached
        //  TODO make this generic template<typename TEventPtr>
        void yield(scheduler::EventPtr<RGTask> event)
        {
            yield_impl<RGTask>(event);
        }

        //! apply a patch to the properties of the currently running task
        void update_properties(
            typename RGTask::TaskProperties::Patch const& patch); // TODO ensure TaskProperties is a TaskProperties1

        //! get backtrace from currently running task
        std::vector<std::reference_wrapper<RGTask>> backtrace() const;

        /*! Create an event on which the termination of the current task depends.
         *  A task must currently be running.
         *
         * @return Handle to flag the event with `reach_event` later.
         *         nullopt if there is no task running currently
         */
        std::optional<scheduler::EventPtr<RGTask>> create_event()
        {
            return create_event_impl<RGTask>();
        }

        unsigned scope_depth() const
        {
            return scope_depth_impl();
        }

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
        template<typename TSchedTag, typename Callable, typename... Args>
        auto emplace_task(Callable&& f, Args&&... args)
        {
            WorkerId worker_id = scheduler_map[TSchedTag{}]->getNextWorkerID();

            SPDLOG_TRACE("emplace task to worker {}", worker_id);

            using Impl = typename std::invoke_result_t<
                BindArgs<Callable, decltype(forward_arg(std::declval<Args>()))...>,
                Callable,
                decltype(forward_arg(std::declval<Args>()))...>;
            // this is not set to nullptr. But it goes out of scope. Memory is managed by allocate
            FunTask<Impl, RGTask>* task;
            memory::Allocator alloc(worker_id);
            memory::Block blk = alloc.allocate(sizeof(FunTask<Impl, RGTask>));
            task = (FunTask<Impl, RGTask>*) blk.ptr;
            SPDLOG_TRACE("Allocated Task of size {}", sizeof(FunTask<Impl, RGTask>));

            if(!task)
                throw std::bad_alloc();

            // construct task in-place
            new(task) FunTask<Impl, RGTask>(worker_id, scope_depth_impl(), *scheduler_map[TSchedTag{}]);

            return TaskBuilder<RGTask, Callable, Args...>(
                task,
                current_task_space(),
                std::forward<Callable>(f),
                std::forward<Args>(args)...);
        }

        template<typename Callable, typename... Args>
        auto emplace_task(Callable&& f, Args&&... args)
        {
            return emplace_task<DefaultTag, Callable, Args...>(std::forward<Callable>(f), std::forward<Args>(args)...);
        }

        template<typename TSchedTag>
        auto& getScheduler()
        {
            return *scheduler_map[TSchedTag{}];
        }

        auto& getScheduler()
        {
            return getScheduler<DefaultTag>();
        }

        template<typename AccessPolicy>
        auto createResource() -> Resource<AccessPolicy>
        {
            return Resource<AccessPolicy>(TaskFreeCtx::create_resource_uid());
        }

    private:
        MapTuple<TSchedMap> scheduler_map;

#if REDGRAPES_ENABLE_TRACE
        std::shared_ptr<perfetto::TracingSession> tracing_session;
#endif
    };

    // TODO make sure init can only be called once
    // require atleast one T execDesc
    template<C_TaskProperty... UserTaskProps, C_Exec T, C_Exec... Ts>
    [[nodiscard]] inline auto init(T execDesc, Ts... execDescs)
    {
        using DescType = boost::mp11::mp_list<T, Ts...>;
        using DescMap = boost::mp11::mp_transform<MakeKeyValList, DescType>;

        return RedGrapes<DescMap, UserTaskProps...>(execDesc, execDescs...);
    }

    template<C_TaskProperty... UserTaskProps>
    [[nodiscard]] inline auto init(WorkerId n_workers = std::thread::hardware_concurrency())
    {
        auto execDesc = SchedulerDescription(
            std::make_shared<scheduler::PoolScheduler<dispatch::thread::DefaultWorker<Task<UserTaskProps...>>>>(
                n_workers),
            DefaultTag{});
        using DescType = boost::mp11::mp_list<decltype(execDesc)>;
        using DescMap = boost::mp11::mp_transform<MakeKeyValList, DescType>;

        return RedGrapes<DescMap, UserTaskProps...>(execDesc);
    }


} // namespace redGrapes

#include "redGrapes/redGrapes.tpp"
