/* Copyright 2021 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <list>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <vector>

#include <moodycamel/concurrentqueue.h>
#include <spdlog/spdlog.h>

#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/task.hpp>

namespace std
{
    using shared_mutex = shared_timed_mutex;
} // namespace std

namespace redGrapes
{
    struct IPrecedenceGraph
    {
        //! stores insert order of all tasks
        std::list< std::shared_ptr<Task> > tasks;
        std::shared_mutex mutex;

        virtual ~IPrecedenceGraph() {}
        virtual void init_dependencies(typename std::list<std::shared_ptr<Task>>::iterator it) = 0;
        virtual void update_dependencies(std::shared_ptr<Task> v) = 0;

        void remove(std::shared_ptr<Task> v)
        {
            std::unique_lock<std::shared_mutex> wrlock(mutex);
            auto it = std::find( std::begin(tasks), std::end(tasks), v );
            if( it != std::end(tasks) )
                tasks.erase(it);
            else
                spdlog::error("try to remove task which is not in list");
        }
    };

    /*! PrecedencePolicy where all vertices are connected
     */
    struct AllSequential
    {
        template<typename T>
        static bool is_serial(T, T)
        {
            return true;
        }
    };

    /*! PrecedencePolicy where no vertex has edges
     */
    struct AllParallel
    {
        template<typename T>
        static bool is_serial(T, T)
        {
            return false;
        }
    };

    template<typename PrecedencePolicy>
    struct PrecedenceGraph : IPrecedenceGraph
    {
        void init_dependencies(typename std::list<std::shared_ptr<Task>>::iterator it)
        {
            //std::shared_lock<std::shared_mutex> rdlock(this->mutex);

            Task & task = *(*it++);
            //SPDLOG_TRACE("PrecedenceGraph::init_dependencies({})", task_vertex->task->task_id);
            for(; it != std::end(this->tasks); ++it)
                if(PrecedencePolicy::is_serial(*(*it), task))
                    task.add_dependency(*it);
        }

        //! remove all edges that are outdated according to PrecedencePolicy
        void update_dependencies( std::shared_ptr<Task> task )
        {
            std::unique_lock<std::shared_mutex> wrlock(task->post_event.followers_mutex);
            auto & out_edges = task->post_event.followers;

            for( unsigned i = 0; i < out_edges.size(); ++i)
            {
                scheduler::EventPtr follower_event = out_edges[i];
                std::shared_ptr<Task> follower = follower_event.task;

                if( !PrecedencePolicy::is_serial(*task, *follower) )
                {
                    std::remove_if(
                       std::begin(follower->in_edges),
                       std::end(follower->in_edges),
                       [task](std::weak_ptr<Task> prev) { return prev.lock() == task; });

                    out_edges.erase(std::next(std::begin(out_edges), i--));

                    follower_event.notify();
                }
            }
        }
    };

    /*!
     */
    struct TaskSpace : std::enable_shared_from_this<TaskSpace>
    {
        moodycamel::ConcurrentQueue<std::shared_ptr<Task>> queue;
        std::shared_ptr<IPrecedenceGraph> precedence_graph;

        std::weak_ptr<Task> parent;

        TaskSpace(
            std::shared_ptr<IPrecedenceGraph> precedence_graph,
            std::weak_ptr<Task> parent = std::weak_ptr<Task>())
            : precedence_graph(precedence_graph)
            , parent(parent)
        {
        }

        /*!
         * takes one task from the queue,
         * inserts it into the precedence graph,
         * calculates the dependencies to its predecessors
         * and creates all edges in the precedence graph accordingly.
         *
         * @return if available, return the task that was
         *         inserted into the precedence graph,
         *         std::nullopt if queue is empty
         */
        std::shared_ptr<Task> next()
        {
            // we need to lock the graph uniquely here to preserve the insertion order,
            // since `try_dequeue()` is lock-free
            std::unique_lock<std::shared_mutex> wrlock(precedence_graph->mutex);

            std::shared_ptr<Task> task;
            if(queue.try_dequeue(task))
            {
                task->space = shared_from_this();

                // insert the new task upfront, so we iterate from latest to earliest.
                auto it = precedence_graph->tasks.insert(std::begin(precedence_graph->tasks), task);
                //wrlock.unlock();

                // from now on, the precedence graph can be shared-locked,
                // since out_edges has its separate mutex
                precedence_graph->init_dependencies(it);

                return task;
            }
            else
                return std::shared_ptr<Task>();
        }

        //! add task to the queue, will be processed lazily by `next()`
        void push(std::shared_ptr<Task> task)
        {
            queue.enqueue(task);
        }

        void remove(std::shared_ptr<Task> v)
        {
            precedence_graph->remove(v);
        }

        bool empty() const
        {
            std::shared_lock<std::shared_mutex> rdlock(precedence_graph->mutex);
            return precedence_graph->tasks.empty() && queue.size_approx() == 0;
        }
    };

} // namespace redGrapes
