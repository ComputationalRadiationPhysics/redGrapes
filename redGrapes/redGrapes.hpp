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
#include <moodycamel/concurrentqueue.h>

#include <redGrapes/task/delayed_functor.hpp>
#include <redGrapes/task/task_result.hpp>
#include <redGrapes/task/task.hpp>

#include <redGrapes/task/property/inherit.hpp>
#include <redGrapes/task/property/trait.hpp>
#include <redGrapes/task/property/id.hpp>
#include <redGrapes/task/property/resource.hpp>

#include <redGrapes/scheduler/scheduling_graph.hpp>
#include <redGrapes/scheduler/default_scheduler.hpp>

namespace redGrapes
{
    template<typename... TaskPropertyPolicies>
    struct TaskBase : TaskProperties<TaskPropertyPolicies...>
    {
        using Props = TaskProperties<TaskPropertyPolicies...>;
        using VertexPtr = std::shared_ptr<PrecedenceGraphVertex<TaskBase<TaskPropertyPolicies...>>>;
        using WeakVertexPtr = std::weak_ptr<PrecedenceGraphVertex<TaskBase<TaskPropertyPolicies...>>>;

        std::shared_ptr<TaskImplBase> impl;

        template<typename F>
        TaskBase(F&& f, TaskProperties<TaskPropertyPolicies...>&& prop)
            : TaskProperties<TaskPropertyPolicies...>(std::move(prop))
            , impl(new FunctorTask<F>(std::move(f)))
        {
        }
    };

    template<typename... TaskPropertyPolicies>
    class RedGrapes : public virtual IManager<TaskBase<IDProperty, ResourceProperty, scheduler::SchedulingGraphProp, scheduler::FIFOSchedulerProp, TaskPropertyPolicies...>>
    {
    public:
        using Task = ::redGrapes::TaskBase<IDProperty, ResourceProperty, scheduler::SchedulingGraphProp, scheduler::FIFOSchedulerProp, TaskPropertyPolicies...>;
        using TaskProps = typename Task::Props;
        using TaskVertexPtr = typename Task::VertexPtr;

    private:
        moodycamel::ConcurrentQueue<TaskVertexPtr> activation_queue;
        moodycamel::ConcurrentQueue<std::shared_ptr<TaskSpace<Task>>> active_task_spaces;

        std::shared_ptr<TaskSpace<Task>> main_space;
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
        RedGrapes( size_t n_threads = std::thread::hardware_concurrency())
            : main_space(std::make_shared<TaskSpace<Task>>(std::make_shared<PrecedenceGraph<Task, ResourceUser>>()))
        {
            active_task_spaces.enqueue(main_space);
            set_scheduler(scheduler::make_default_scheduler<Task>(*this, n_threads));
        }

        ~RedGrapes()
        {
            while( ! main_space->empty() )
            {
                SPDLOG_TRACE("RedGrapes: idle");
                redGrapes::thread::idle();
            }

            SPDLOG_TRACE("RedGrapes: scheduling graph empty!");

            scheduler->notify();
        }

        /*! Initialize the scheduler to work with this manager.
         * Must be called at initialization before any call to `emplace_task`.
         */
        void set_scheduler(std::shared_ptr<scheduler::IScheduler<Task>> scheduler)
        {
            this->scheduler = scheduler;
            //this->scheduler->start();
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
        auto emplace_task(Callable&& f, typename TaskProps::Builder builder, Args&&... args)
        {
            PropBuildHelper build_helper{builder};
            pass(build_helper.template build<Args>(args)...);

            build_helper.foo();

            auto impl = std::bind(f, std::forward<Args>(args)...);

            auto delayed = make_delayed_functor(std::move(impl));
            auto future = delayed.get_future();

            builder.init_id(); // needed because property builder may be copied from a template

            auto result_event = std::make_shared< scheduler::Event >();
            auto task = std::make_unique<Task>(
                std::bind(
                    [this, result_event](auto&& delayed) mutable
                    {
                        delayed();
                        result_event->reach();
                        scheduler->notify();
                    },
                    std::move(delayed)),
                std::move(builder));

            if( auto parent = current_task() )
                task->impl->scope_level = (*parent)->task->impl->scope_level + 1;
            else
                task->impl->scope_level = 1;

            SPDLOG_DEBUG("RedGrapes::emplace_task {}\n", (TaskProps const&) *task);

            current_task_space()->push(std::move(task));
            scheduler->notify();

            return make_task_result(std::move(future), *this, result_event);
        }

        std::optional<TaskVertexPtr>& current_task()
        {
            static thread_local std::optional<TaskVertexPtr> current_task;
            return current_task;
        }

        //! enqueue task in activation queue
        void activate_task(TaskVertexPtr vertex_ptr)
        {
            SPDLOG_TRACE("mgr: add task {} to activation queue", vertex_ptr->task->task_id);

            if(!vertex_ptr->task->in_activation_queue.test_and_set())
            {
                activation_queue.enqueue(vertex_ptr);
                scheduler->notify();
            }
        }

        //! push next task from activation queue to scheduler, if available
        bool activate_next()
        {
            TaskVertexPtr vertex_ptr;
            if( activation_queue.try_dequeue(vertex_ptr) )
            {
                vertex_ptr->task->in_activation_queue.clear();
                scheduler->activate_task(vertex_ptr);
                return true;
            }
            else
                return false;
        }

        std::shared_ptr<TaskSpace<Task>> current_task_space()
        {
            if(auto task = current_task())
            {
                if((*task)->children == std::nullopt)
                {
                    auto task_space = std::make_shared<TaskSpace<Task>>(
                        std::make_shared<PrecedenceGraph<Task, ResourceUser>>(),
                        std::weak_ptr<PrecedenceGraphVertex<Task>>(*task));

                    active_task_spaces.enqueue(task_space);

                    (*task)->children = task_space;
                }

                return *(*task)->children;
            }
            else
                return main_space;
        }

        void notify_event( std::shared_ptr< scheduler::Event > event )
        {
            
        }

        void init_new_task( TaskVertexPtr task_ptr )
        {
            task_ptr->task->ready_hook =
                [this, task_ptr]
                {
                    activate_task(task_ptr);
                    //scheduler->notify();
                };

            task_ptr->task->init_scheduling_graph(
                task_ptr,
                [this, task_ptr]
                {
                    if(auto children = task_ptr->children)
                        while(auto new_task = (*children)->next())
                            init_new_task( task_ptr );
                    else
                        remove_task(task_ptr);
                });

            // maybe the new task is ready
            auto pe = task_ptr->task->pre_event;
            if(pe)
                pe->notify();

            scheduler->notify();
        }

        void update_active_task_spaces()
        {
            std::vector< std::shared_ptr< TaskSpace<Task> > > buf;

            std::shared_ptr< TaskSpace<Task> > space;
            while(active_task_spaces.try_dequeue(space))
            {
                while(auto new_task = space->next())
                    init_new_task(*new_task);

                bool remove = false;
                if( auto parent_weak = space->parent )
                {
                    auto parent_vertex = parent_weak->lock();
                    Task & parent_task = *parent_vertex->task;
                    if(
                        space->empty() &&
                        parent_task.is_finished()
                    )
                    {
                        remove_task(parent_vertex);
                        remove = true;
                    }
                }

                if(! remove)
                    buf.push_back(space);
            }

            for( auto space : buf )
                active_task_spaces.enqueue(space);
        }

        std::shared_ptr<TaskSpace<Task>> get_main_space()
        {
            return main_space;
        }

        std::shared_ptr< scheduler::IScheduler<Task> > get_scheduler()
        {
            return scheduler;
        }

        void remove_task(TaskVertexPtr task_vertex)
        {
            if( auto task_space = task_vertex->space.lock() )
                task_space->remove(task_vertex);

            // drop this to break cycles
            task_vertex->task->ready_hook = std::function<void()>();
            task_vertex->task->pre_event = nullptr;
            task_vertex->task->post_event = nullptr;
        }

        /*! Get the TaskID of the currently running task.
         * @return nullopt if there is no task running currently.
         */
        std::optional<TaskID> get_current_task_id()
        {
            if(auto task_vertex = current_task())
                return (*task_vertex)->task->task_id;
            else
                return std::experimental::nullopt;
        }

        /*! Create an event on which the termination of the current task depends.
         *  A task must currently be running.
         *
         * @return Handle to flag the event with `reach_event` later.
         *         nullopt if there is no task running currently
         */
        std::optional< std::shared_ptr<scheduler::Event> > create_event()
        {
            if(auto task_ptr = current_task())
            {
                auto event = (*task_ptr)->task->make_event();
                event->reach_hook =
                    [this]
                    {
                        scheduler->notify();
                    };

                return event;
            }
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

        /*! wait until all tasks finished
         * can only be called outside of a task
         */
        void wait_for_all()
        {
            SPDLOG_TRACE("wait for all tasks...");
            if(!current_task())
                while(!main_space->empty())
                    thread::idle();
            else
                throw std::runtime_error("called wait_for_all() inside a task!");
        }

        //! pause the currently running task at least until event_id is reached
        void yield( std::shared_ptr<scheduler::Event> event )
        {
            while(! event->is_reached() )
            {
                if(auto cur_vertex = current_task())
                    (*cur_vertex)->task->impl->yield(event);
                else
                    thread::idle();
            }
        }

        //! get backtrace from currently running task
        std::vector<TaskProps> backtrace()
        {
            std::vector<TaskProps> bt;
            std::optional<TaskVertexPtr> task_vertex = current_task();

            while( task_vertex )
            {
                bt.push_back(*(*task_vertex)->task);

                if( auto parent = (*task_vertex)->space.lock()->parent )
                    task_vertex = (*parent).lock();
                else
                    task_vertex = std::nullopt;
            }

            return bt;
        }
    };

} // namespace redGrapes

