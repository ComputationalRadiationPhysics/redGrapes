/* Copyright 2019-2021 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <shared_mutex>
#include <unordered_map>
#include <sstream>

#include <spdlog/spdlog.h>

#include <redGrapes/graph/scheduling_graph.hpp>

#include <redGrapes/task/delayed_functor.hpp>
#include <redGrapes/task/task_result.hpp>
#include <redGrapes/task/task.hpp>

#include <redGrapes/property/inherit.hpp>
#include <redGrapes/property/trait.hpp>
#include <redGrapes/property/id.hpp>
#include <redGrapes/property/resource.hpp>

#include <redGrapes/scheduler/default_scheduler.hpp>

namespace redGrapes
{
    template<typename... TaskPropertyPolicies>
    struct TaskBase : TaskProperties<TaskPropertyPolicies...>
    {
        std::shared_ptr<TaskImplBase> impl;

        template<typename F>
        TaskBase(F&& f, TaskProperties<TaskPropertyPolicies...>&& prop)
            : TaskProperties<TaskPropertyPolicies...>(std::move(prop))
            , impl(new FunctorTask<F>(std::move(f)))
        {
        }
    };

    template<typename... TaskPropertyPolicies>
    class Manager : virtual IManager<TaskBase<IDProperty, ResourceProperty, TaskPropertyPolicies...>>
    {
    public:
        using TaskProps = TaskProperties<IDProperty, ResourceProperty, TaskPropertyPolicies...>;
        using Task = ::redGrapes::TaskBase<IDProperty, ResourceProperty, TaskPropertyPolicies...>;

        using TaskVertexPtr = std::shared_ptr<PrecedenceGraphVertex<Task>>;

    private:
        std::shared_ptr<TaskSpace<Task>> main_space;
        std::shared_ptr<SchedulingGraph<Task>> scheduling_graph;
        std::shared_ptr<scheduler::IScheduler<Task>> scheduler;

        template<typename... Args>
        static inline void pass(Args&&...)
        {
        }

        struct PropBuildHelper
        {
            typename TaskProps::Builder& builder;

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

    public:
        Manager()
            : main_space(std::make_shared<TaskSpace<Task>>(std::make_shared<PrecedenceGraph<Task, ResourceUser>>()))
            , scheduling_graph(std::make_shared<SchedulingGraph<Task>>(
                  [this](TaskVertexPtr a, TaskVertexPtr b) { return this->scheduler->task_dependency_type(a, b); }))
        {
            set_scheduler(scheduler::make_default_scheduler<Task>(*this));
        }

        ~Manager()
        {
            while( ! scheduling_graph->empty() )
            {
                spdlog::trace("Manager: idle");
                redGrapes::thread::idle();
            }

            spdlog::trace("Manager: scheduling graph empty!");

            scheduler->notify();
        }

        /*! Initialize the scheduler to work with this manager.
         * Must be called at initialization before any call to `emplace_task`.
         */
        void set_scheduler(std::shared_ptr<scheduler::IScheduler<Task>> scheduler)
        {
            this->scheduler = scheduler;
        }

        std::shared_ptr< SchedulingGraph<Task> > get_scheduling_graph()
        {
            return std::shared_ptr<SchedulingGraph<Task>>(scheduling_graph);
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
        template<typename Callable, typename... Args>
        auto emplace_task(Callable&& f, Args&&... args)
        {
            typename TaskProps::Builder builder;
            return emplace_task(f, builder, std::forward<Args>(args)...);
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
        auto emplace_task(Callable&& f, typename TaskProps::Builder builder, Args&&... args)
        {
            PropBuildHelper build_helper{builder};
            pass(build_helper.template build<Args>(args)...);

            build_helper.foo();

            auto impl = std::bind(f, std::forward<Args>(args)...);

            auto delayed = make_delayed_functor(std::move(impl));
            auto future = delayed.get_future();

            EventID result_event = scheduling_graph->new_event();
            builder.init_id();
            auto task = std::make_unique<Task>(
                std::bind(
                    [this, result_event](auto&& delayed) mutable
                    {
                        delayed();
                        reach_event(result_event);
                    },
                    std::move(delayed)),
                std::move(builder));

            spdlog::debug("Manager::emplace_task {}\n", (TaskProps const&) *task);
            spdlog::trace("Manager: result_event = {}", result_event);

            current_task_space()->push(std::move(task));
            scheduler->notify();

            return make_task_result(std::move(future), *this, result_event);
        }

        static std::optional<TaskVertexPtr>& current_task()
        {
            static thread_local std::optional<TaskVertexPtr> current_task;
            return current_task;
        }

        std::shared_ptr<TaskSpace<Task>> current_task_space()
        {
            if(auto task = current_task())
            {
                if( (*task)->children == std::nullopt )
                    (*task)->children = std::make_shared<TaskSpace<Task>>(std::make_shared<PrecedenceGraph<Task, ResourceUser>>());

                return *(*task)->children;
            }
            else
                return main_space;
        }

        /*! Start the execution of a task.
         *
         * @return true if the task finished, false if it was paused.
         */
        bool run_task(TaskVertexPtr task_vertex)
        {
            spdlog::debug("Manager::run_task({})", task_vertex->task->task_id);

            current_task() = task_vertex;
            bool finished = (*task_vertex->task->impl)();
            current_task() = std::nullopt;

            return finished;
        }

        void activate_task(TaskVertexPtr task_vertex)
        {
            spdlog::trace("Manager::activate_task({})", task_vertex->task->task_id);
            scheduler->activate_task(task_vertex);
        }

        void remove_task(TaskVertexPtr task_vertex)
        {
            task_vertex->space.lock()->remove(task_vertex);
            scheduling_graph->remove_task(task_vertex->task->task_id);
        }

        /*! Get the TaskID of the currently running task.
         * @return nullopt if there is no task running currently.
         */
        std::optional<TaskID> get_current_task_id()
        {
            if(auto task_vertex = current_task())
                return task_vertex->task->task_id;
            else
                return std::experimental::nullopt;
        }

        //! flag the state of the event & update
        void reach_event(EventID event_id)
        {
            scheduling_graph->reach_event(event_id);
            scheduler->notify();
        }

        /*! Create an event on which the termination of the current task depends.
         *  A task must currently be running.
         *
         * @return Handle to flag the event with `reach_event` later.
         *         nullopt if there is no task running currently
         */
        std::optional<EventID> create_event()
        {
            if(auto task_id = get_current_task_id())
                return scheduling_graph->add_post_dependency(*task_id);
            else
                return std::nullopt;
        }

        //! apply a patch to the properties of the currently running task
        void update_properties(typename TaskProps::Patch const& patch)
        {
            if(auto task_vertex = current_task())
            {
                task_vertex->task->apply_patch(patch);
                /* TODO!
                auto vertices = task_ptr->graph->update_vertex(task_ptr->vertex);

                scheduling_graph->update_task(*task_ptr, followers);

                for(auto following_task : followers)
                    scheduler->activate_task(following_task);

                lock.unlock();

                scheduler->notify();
                */
            }
            else
                throw std::runtime_error("update_properties: currently no task running");
        }

        void notify()
        {
            scheduler->notify();
        }
        
        /*! wait until all tasks finished
         * can only be called outside of a task
         */
        void wait_for_all()
        {
            spdlog::trace("wait for all tasks...");
            if(!current_task())
                while(!scheduling_graph->empty())
                    thread::idle();
            else
                throw std::runtime_error("called wait_for_all() inside a task!");
        }

        //! pause the currently running task at least until event_id is reached
        void yield(EventID event_id)
        {
            while(!scheduling_graph->is_event_reached(event_id))
            {
                if(auto cur_vertex = current_task())
                {
                    auto& task = *(*cur_vertex)->task;
                    spdlog::trace("pause task {}", task.task_id);
                    scheduling_graph->task_pause(task.task_id, event_id);
                    task.impl->yield();
                }
                else
                    thread::idle();
            }
        }

        //! get backtrace from currently running task
        std::vector<TaskProps> backtrace()
        {
            std::vector<TaskProps> bt;
            std::optional<TaskVertexPtr> task_vertex = current_task();

            while(task_vertex)
            {
                bt.push_back(*(*task_vertex)->task);

                if( auto parent = (*task_vertex)->space.lock()->parent )
                    task_vertex = parent->lock();
                else
                    task_vertex = std::nullopt;
            }

            return bt;
        }
    };

} // namespace redGrapes

