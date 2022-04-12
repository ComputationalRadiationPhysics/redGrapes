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
#include <future>

#include <spdlog/spdlog.h>
#include <moodycamel/concurrentqueue.h>

#include <redGrapes/task/delayed_functor.hpp>
#include <redGrapes/task/task_result.hpp>

#include <redGrapes/task/task.hpp>
#include <redGrapes/task/property/id.hpp>
#include <redGrapes/task/property/resource.hpp>

#include <redGrapes/scheduler/scheduling_graph.hpp>
#include <redGrapes/scheduler/default_scheduler.hpp>
#include <redGrapes/dispatch/dispatcher.hpp>

namespace redGrapes
{
    template<
        typename... TaskPropertyPolicies
    >
    class RedGrapes : public virtual IManager
    {
    public:
        using Task = task::PropTask<
            IDProperty,
            ResourceProperty,
            scheduler::SchedulingGraphProp,
            TaskPropertyPolicies...
        >;
        using TaskProps = typename Task::Props;
        using TaskVertexPtr = typename Task::VertexPtr;

    private:
        moodycamel::ConcurrentQueue<TaskVertexPtr> activation_queue;
        moodycamel::ConcurrentQueue<std::shared_ptr<TaskSpace>> active_task_spaces;

        std::shared_ptr<TaskSpace> main_space;
        std::shared_ptr<scheduler::IScheduler> scheduler;

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
            : main_space(std::make_shared<TaskSpace>(std::make_shared<PrecedenceGraph<Task, ResourceUser>>()))
        {
            active_task_spaces.enqueue(main_space);
            set_scheduler(scheduler::make_default_scheduler<Task>(*this, n_threads));
        }

        ~RedGrapes()
        {
            while( ! main_space->empty() )
            {
                SPDLOG_TRACE("RedGrapes: idle");
                redGrapes::dispatch::thread::idle();
            }

            SPDLOG_TRACE("RedGrapes: scheduling graph empty!");

            scheduler->notify();
        }

        /*! Initialize the scheduler to work with this manager.
         * Must be called at initialization before any call to `emplace_task`.
         */
        void set_scheduler(std::shared_ptr<scheduler::IScheduler> scheduler)
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

            /* todo: std::packaged_task could replace the internal DelayedFunctor,
             * but how exactly ?
             */
            //std::packaged_task< typename std::result_of< Callable() >::type() > delayed(std::move(impl));

            auto delayed = make_delayed_functor(std::move(impl));
            auto future = delayed.get_future();

            builder.init_id(); // needed because property builder may be copied from a template

            auto result_event = std::make_shared< scheduler::Event >();

            auto task = task::make_fun_task(
                std::bind(
                    [this, result_event](auto&& delayed) mutable
                    {                        
                        delayed();
                        this->notify_event(result_event);
                    },
                    std::move(delayed)),
                (TaskProps)builder);

            if( auto parent = current_task() )
                task->scope_level = (*parent)->template get_task<Task>().scope_level + 1;
            else
                task->scope_level = 1;

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
            SPDLOG_TRACE("mgr: add task {} to activation queue", vertex_ptr->template get_task<Task>().task_id);
            //activation_queue.enqueue(vertex_ptr);
            scheduler->activate_task(vertex_ptr);
        }
        /*
        //! push next task from activation queue to scheduler, if available
        bool activate_next()
        {
            SPDLOG_TRACE("activate_next()");
            TaskVertexPtr task_vertex;
            if( activation_queue.try_dequeue(task_vertex) )
            {
                //task_vertex->task->in_activation_queue.clear();
                scheduler->activate_task(task_vertex);
                return true;
            }
            else
                return false;
        }
        */

        std::shared_ptr<TaskSpace> current_task_space()
        {
            if(auto task = current_task())
            {
                if((*task)->children == std::nullopt)
                {
                    auto task_space = std::make_shared<TaskSpace>(
                        std::make_shared<PrecedenceGraph<Task, ResourceUser>>(),
                        std::weak_ptr<PrecedenceGraphVertex>(*task));

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
            event->notify(
                [this]( int state, std::shared_ptr< scheduler::Event > event )
                {
                    SPDLOG_TRACE("notify event {} state={}", (void*)event.get(), state);

                    auto weak_task_vertex = event->task_vertex;
                    auto task_vertex = weak_task_vertex.lock();

                    if( task_vertex )
                    {
                        Task & task = task_vertex->get_task<Task>();

                        // pre event ready
                        if( event == task.pre_event && state == 1 )
                        {
                            SPDLOG_TRACE("pre event ready");
                            this->activate_task(task_vertex);
                        }

                        // post event reached
                        if( event == task.post_event && state == 0 )
                        {
                            SPDLOG_TRACE("post event reached");
                            if(auto children = task_vertex->children)
                                while(auto new_task = (*children)->next())
                                {
                                    auto & task = (*new_task)->get_task<Task>();
                                    task.template sg_init<Task>(*this, *new_task);

                                    task.pre_event->up();
                                    notify_event( task.pre_event );
                                }
                            else
                                this->remove_task(task_vertex);
                        }
                    }

                    this->scheduler->notify();
                });

            this->scheduler->notify();
        }

        void update_active_task_spaces()
        {
            SPDLOG_TRACE("update active task spaces");
            std::vector< std::shared_ptr< TaskSpace > > buf;

            std::shared_ptr< TaskSpace > space;
            while(active_task_spaces.try_dequeue(space))
            {
                while(auto new_task = space->next())
                {
                    auto & task = (*new_task)->get_task<Task>();
                    task.template sg_init<Task>(*this, *new_task);

                    task.pre_event->up();
                    notify_event( task.pre_event );
                }

                bool remove = false;
                if( auto parent_weak = space->parent )
                {
                    auto parent_vertex = parent_weak->lock();
                    Task & parent_task = parent_vertex->get_task<Task>();
                    if(
                        space->empty()
                        && parent_task.is_finished()
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

        std::shared_ptr<TaskSpace> get_main_space()
        {
            return main_space;
        }

        std::shared_ptr< scheduler::IScheduler > get_scheduler()
        {
            return scheduler;
        }

        void remove_task(TaskVertexPtr task_vertex)
        {
            SPDLOG_TRACE("remove task {}", task_vertex->template get_task<Task>().task_id);
            if( auto task_space = task_vertex->space.lock() )
            {
                task_space->remove(task_vertex);
                scheduler->notify();
            }
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
                return (*task_ptr)->template get_task<Task>().make_event();
            else
                return std::nullopt;
        }

        //! apply a patch to the properties of the currently running task
        void update_properties(typename TaskProps::Patch const& patch)
        {
            if(auto task_vertex = current_task())
            {
                auto & task = (*task_vertex)->template get_task<Task>();
                task.apply_patch(patch);

                std::vector< TaskVertexPtr > revoked_vertices = current_task_space()->precedence_graph->update_dependencies( *task_vertex );
                task.template sg_revoke_followers<Task>( *this, *task_vertex, revoked_vertices );
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
                    dispatch::thread::idle();
            else
                throw std::runtime_error("called wait_for_all() inside a task!");
        }

        //! pause the currently running task at least until event_id is reached
        void yield( std::shared_ptr<scheduler::Event> event )
        {
            while(! event->is_reached() )
            {
                if(auto cur_vertex = current_task())
                    (*cur_vertex)->template get_task<Task>().yield(event);
                else
                    dispatch::thread::idle();
            }
        }

        //! get backtrace from currently running task
        std::vector<TaskProps> backtrace()
        {
            std::vector<TaskProps> bt;
            std::optional<TaskVertexPtr> task_vertex = current_task();

            while( task_vertex )
            {
                bt.push_back((*task_vertex)->template get_task<Task>());

                if( auto parent = (*task_vertex)->space.lock()->parent )
                    task_vertex = (*parent).lock();
                else
                    task_vertex = std::nullopt;
            }

            return bt;
        }
    };

} // namespace redGrapes

