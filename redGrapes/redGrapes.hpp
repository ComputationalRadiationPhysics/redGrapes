/* Copyright 2022-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/SchedulerDescription.hpp"
#include "redGrapes/TaskCtx.hpp"
#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/memory/hwloc_alloc.hpp"
#include "redGrapes/resource/fieldresource.hpp"
#include "redGrapes/resource/ioresource.hpp"
#include "redGrapes/scheduler/event.hpp"
#include "redGrapes/scheduler/pool_scheduler.hpp"
#include "redGrapes/task/task.hpp"
#include "redGrapes/task/task_space.hpp"
#include "redGrapes/util/bind_args.hpp"
#include "redGrapes/util/tuple_map.hpp"

#include <boost/mp11.hpp>
#include <spdlog/spdlog.h>

#include <memory>
#include <new>

// `TaskBuilder` needs "RedGrapes`, so can only include here after definiton
#include "redGrapes/task/task_builder.hpp" // TODO change this to needs LocalImpl

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

            TaskFreeCtx::n_pus = hwloc_get_nbobjs_by_type(TaskFreeCtx::hwloc_ctx.topology, HWLOC_OBJ_PU);
            if(TaskFreeCtx::n_workers > TaskFreeCtx::n_pus)
                spdlog::warn(
                    "{} worker-threads requested, but only {} PUs available!",
                    TaskFreeCtx::n_workers,
                    TaskFreeCtx::n_pus);

            TaskFreeCtx::worker_alloc_pool = std::make_shared<WorkerAllocPool>();
            TaskFreeCtx::worker_alloc_pool->allocs.reserve(TaskFreeCtx::n_workers);

            TaskCtx<RGTask>::root_space = std::make_shared<TaskSpace<RGTask>>();

            auto initAdd = [](auto scheduler, auto& base_worker_id)
            {
                scheduler->init(base_worker_id);
                base_worker_id = base_worker_id + scheduler->n_workers;
            };
            unsigned base_worker_id = 0;
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
            TaskCtx<RGTask>::root_space.reset();

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
            TaskCtx<RGTask>::yield(event);
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
            return TaskCtx<RGTask>::create_event();
        }

        unsigned scope_depth() const
        {
            return TaskCtx<RGTask>::scope_depth();
        }

        std::shared_ptr<TaskSpace<RGTask>> current_task_space() const
        {
            return TaskCtx<RGTask>::current_task_space();
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

            SPDLOG_TRACE("emplace task to worker {} next_worker={}", worker_id, TaskFreeCtx::next_worker);

            using Impl = typename std::invoke_result_t<BindArgs<Callable, Args...>, Callable, Args...>;
            // this is not set to nullptr. But it goes out of scope. Memory is managed by allocate
            FunTask<Impl, RGTask>* task;
            memory::Allocator alloc(worker_id);
            memory::Block blk = alloc.allocate(sizeof(FunTask<Impl, RGTask>));
            task = (FunTask<Impl, RGTask>*) blk.ptr;

            if(!task)
                throw std::bad_alloc();

            // construct task in-place
            new(task) FunTask<Impl, RGTask>(*scheduler_map[TSchedTag{}]);

            task->worker_id = worker_id;

            return std::move(TaskBuilder<RGTask, Callable, Args...>(task, std::move(f), std::forward<Args>(args)...));
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

        template<typename Container>
        auto createFieldResource(Container* c) -> FieldResource<Container, RGTask>
        {
            return FieldResource<Container, RGTask>(c);
        }

        template<typename Container, typename... Args>
        auto createFieldResource(Args&&... args) -> FieldResource<Container, RGTask>
        {
            return FieldResource<Container, RGTask>(args...);
        }

        template<typename T>
        auto createIOResource(std::shared_ptr<T> o) -> IOResource<T, RGTask>
        {
            return IOResource<T, RGTask>(o);
        }

        template<typename T, typename... Args>
        auto createIOResource(Args&&... args) -> IOResource<T, RGTask>
        {
            return IOResource<T, RGTask>(args...);
        }

        template<typename AccessPolicy>
        auto createResource() -> Resource<RGTask, AccessPolicy>
        {
            return Resource<RGTask, AccessPolicy>();
        }

    private:
        MapTuple<TSchedMap> scheduler_map;
    };

    // TODO make sure init can only be called once
    template<C_TaskProperty... UserTaskProps, C_Exec... Ts>
    [[nodiscard]] inline auto init(Ts... execDescs)
    {
        using DescType = boost::mp11::mp_list<Ts...>;
        using DescMap = boost::mp11::mp_transform<MakeKeyValList, DescType>;

        return RedGrapes<DescMap, UserTaskProps...>(execDescs...);
    }

    template<C_TaskProperty... UserTaskProps>
    [[nodiscard]] inline auto init(size_t n_workers = std::thread::hardware_concurrency())
    {
        auto execDesc = SchedulerDescription(
            std::make_shared<scheduler::PoolScheduler<
                Task<UserTaskProps...>,
                dispatch::thread::DefaultWorker<Task<UserTaskProps...>>>>(n_workers),
            DefaultTag{});
        using DescType = boost::mp11::mp_list<decltype(execDesc)>;
        using DescMap = boost::mp11::mp_transform<MakeKeyValList, DescType>;

        return RedGrapes<DescMap, UserTaskProps...>(execDesc);
    }


} // namespace redGrapes

#include "redGrapes/dispatch/thread/DefaultWorker.tpp"
#include "redGrapes/dispatch/thread/worker_pool.tpp"
#include "redGrapes/redGrapes.tpp"
#include "redGrapes/resource/resource_user.tpp"
#include "redGrapes/scheduler/event.tpp"
#include "redGrapes/scheduler/event_ptr.tpp"
#include "redGrapes/scheduler/pool_scheduler.tpp"
#include "redGrapes/task/property/graph.tpp"
#include "redGrapes/util/trace.tpp"
