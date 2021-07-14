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

namespace std
{
    using shared_mutex = shared_timed_mutex;
} // namespace std

namespace redGrapes
{
    template<typename Task>
    struct PrecedenceGraphVertex;

    template<typename Task>
    struct ITaskSpace
    {
        using TaskVertexPtr = std::shared_ptr<PrecedenceGraphVertex<Task>>;

        virtual ~ITaskSpace() = 0;

        virtual std::optional<TaskVertexPtr> next() = 0;
        virtual void push_task(std::unique_ptr<Task>&& task);
    };

    template<typename Task>
    struct TaskSpace;

    template<typename Task>
    struct PrecedenceGraphVertex
    {
        std::unique_ptr<Task> task;

        std::weak_ptr<TaskSpace<Task>> space;
        std::optional<std::shared_ptr<TaskSpace<Task>>> children;

        // out_edges needs a mutex because edges will be added later on
        std::vector<std::weak_ptr<PrecedenceGraphVertex<Task>>> out_edges;
        std::shared_mutex out_edges_mutex;

        // in edges dont need a mutex because they are initialized
        // once by `init_dependencies()` and only read afterwards.
        // expired pointers must be ignored
        std::vector<std::weak_ptr<PrecedenceGraphVertex<Task>>> in_edges;

        PrecedenceGraphVertex(std::unique_ptr<Task>&& task, std::weak_ptr<TaskSpace<Task>> space)
            : task(std::move(task))
            , space(space)
        {
        }
    };

    template<typename Task>
    struct IPrecedenceGraph
    {
        using TaskVertexPtr = std::shared_ptr<PrecedenceGraphVertex<Task>>;

        //! stores insert order of all tasks
        std::list<TaskVertexPtr> tasks;
        std::shared_mutex mutex;

        virtual ~IPrecedenceGraph()
        {
        }
        virtual void init_dependencies(typename std::list<TaskVertexPtr>::iterator it)
        {
        }
        virtual void update_dependencies(TaskVertexPtr v)
        {
        }

        void remove(TaskVertexPtr v)
        {
            std::unique_lock<std::shared_mutex> wrlock(mutex);
            tasks.remove(v);
        }

        void add_edge(TaskVertexPtr u, TaskVertexPtr v)
        {
            v->in_edges.push_back(u);

            std::unique_lock<std::shared_mutex> wrlock(u->out_edges_mutex);
            u->out_edges.push_back(v);
        }
    };

    /*! EnqueuePolicy where all vertices are connected
     */
    struct AllSequential
    {
        template<typename T>
        static bool is_serial(T, T)
        {
            return true;
        }
    };

    /*! EnqueuePolicy where no vertex has edges
     */
    struct AllParallel
    {
        template<typename T>
        static bool is_serial(T, T)
        {
            return false;
        }
    };

    template<typename Task, typename EnqueuePolicy>
    struct PrecedenceGraph : IPrecedenceGraph<Task>
    {
        using TaskVertexPtr = std::shared_ptr<PrecedenceGraphVertex<Task>>;

        void init_dependencies(typename std::list<TaskVertexPtr>::iterator it)
        {
            std::shared_lock<std::shared_mutex> rdlock(this->mutex);

            TaskVertexPtr task_vertex = *it;
            for(; it != std::end(this->tasks); ++it)
                if(EnqueuePolicy::is_serial(*(*it)->task, *task_vertex->task))
                    this->add_edge(*it, task_vertex);
        }

        void update_dependencies(TaskVertexPtr v)
        {
            /*
            std::unique_lock<std::shared_mutex> wrlock(v->out_edges_mutex);
            std::remove_if(
                std::begin(v->out_edges),
                std::end(v->out_edges),
                [v](TaskVertexPtr w) { return !EnqueuePolicy::is_serial(*v->task, *w->task); });
            */
            // todo: in edges?
            // todo: update scheduling graph
        }
    };

    /*!
     */
    template<typename Task>
    struct TaskSpace : std::enable_shared_from_this<TaskSpace<Task>>
    {
        moodycamel::ConcurrentQueue<std::unique_ptr<Task>> queue;
        std::shared_ptr<IPrecedenceGraph<Task>> precedence_graph;

        std::optional<std::weak_ptr<PrecedenceGraphVertex<Task>>> parent;

        using TaskVertexPtr = std::shared_ptr<PrecedenceGraphVertex<Task>>;

        TaskSpace(std::shared_ptr<IPrecedenceGraph<Task>> precedence_graph) : precedence_graph(precedence_graph)
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
        std::optional<TaskVertexPtr> next()
        {
            // we need to lock the graph uniquely here to preserve the insertion order,
            // since `try_dequeue()` is lock-free
            std::unique_lock<std::shared_mutex> wrlock(precedence_graph->mutex);

            std::unique_ptr<Task> task;
            if(queue.try_dequeue(task))
            {
                // insert the new task upfront, so we iterate from latest to earliest.
                auto it = precedence_graph->tasks.insert(
                    std::begin(precedence_graph->tasks),
                    std::make_shared<PrecedenceGraphVertex<Task>>(std::move(task), this->shared_from_this()));
                wrlock.unlock();

                // from now on, the precedence graph can be shared-locked,
                // since out_edges has its separate mutex
                precedence_graph->init_dependencies(it);
                return *it;
            }
            else
                return std::nullopt;
        }

        //! add task to the queue, will be processed lazily by `next()`
        void push(std::unique_ptr<Task>&& task)
        {
            queue.enqueue(std::move(task));
        }

        void remove(TaskVertexPtr v)
        {
            precedence_graph->remove(v);
        }
    };

} // namespace redGrapes
