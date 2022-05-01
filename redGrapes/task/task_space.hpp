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

//#include <redGrapes/graph/scheduling_graph.hpp>

namespace std
{
    using shared_mutex = shared_timed_mutex;
} // namespace std

namespace redGrapes
{
    struct PrecedenceGraphVertex;
    struct TaskSpace;

    using TaskVertexPtr = std::shared_ptr< PrecedenceGraphVertex >;

    struct ITaskSpace
    {
        virtual ~ITaskSpace() = 0;

        virtual std::optional<TaskVertexPtr> next() = 0;
        virtual void push(std::unique_ptr<ITask>&& task) = 0;
        virtual void remove(TaskVertexPtr v) = 0;
    };

    struct PrecedenceGraphVertex
    {
        std::unique_ptr<ITask> task;

        std::weak_ptr< TaskSpace > space;
        std::optional<std::shared_ptr<TaskSpace>> children;

        // out_edges needs a mutex because edges will be added later on
        std::vector<std::weak_ptr<PrecedenceGraphVertex>> out_edges;
        std::shared_mutex out_edges_mutex;

        // in edges dont need a mutex because they are initialized
        // once by `init_dependencies()` and only read afterwards.
        // expired pointers must be ignored
        std::vector<std::weak_ptr<PrecedenceGraphVertex>> in_edges;

        PrecedenceGraphVertex(std::unique_ptr<ITask>&& task, std::weak_ptr<TaskSpace> space)
            : task(std::move(task))
            , space(space)
            , children(std::nullopt)
        {
        }

        template <typename Task>
        Task & get_dyn_task()
        {
            return dynamic_cast<Task&>(*task);
        }

        template <typename Task>
        Task & get_task()
        {
            // use dynamic_cast for debug
            return reinterpret_cast<Task&>(*task);
        }
    };

    struct IPrecedenceGraph
    {
        //! stores insert order of all tasks
        std::list<TaskVertexPtr> tasks;
        std::shared_mutex mutex;

        virtual ~IPrecedenceGraph()
        {
            //std::shared_lock<std::shared_mutex> rdlock(this->mutex);
            //assert(tasks.empty());
        }

        virtual void init_dependencies(typename std::list<TaskVertexPtr>::iterator it) = 0;
        virtual std::vector< TaskVertexPtr > update_dependencies(TaskVertexPtr v) = 0;
        void remove(TaskVertexPtr v)
        {
            std::unique_lock<std::shared_mutex> wrlock(mutex);

            auto it = std::find( std::begin(tasks), std::end(tasks), v );
            if( it != std::end(tasks) )
            {
                tasks.erase(it);
            }
            else
            {
                spdlog::error("try to remove task which is not in list");
            }
        }

        void add_edge(TaskVertexPtr u, TaskVertexPtr v)
        {
            //SPDLOG_TRACE("TaskSpace: add edge task {} -> task {}", u->task->task_id, v->task->task_id);
            v->in_edges.push_back(u);

            std::unique_lock<std::shared_mutex> wrlock(u->out_edges_mutex);
            u->out_edges.push_back(v);
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

    template<typename Task, typename PrecedencePolicy>
    struct PrecedenceGraph : IPrecedenceGraph
    {
        void init_dependencies(typename std::list<TaskVertexPtr>::iterator it)
        {
            //std::shared_lock<std::shared_mutex> rdlock(this->mutex);

            TaskVertexPtr task_vertex = *it++;
            //SPDLOG_TRACE("PrecedenceGraph::init_dependencies({})", task_vertex->task->task_id);
            for(; it != std::end(this->tasks); ++it)
                if(PrecedencePolicy::is_serial((*it)->template get_task<Task>(), task_vertex->template get_task<Task>()))
                    this->add_edge(*it, task_vertex);
        }

        //! remove all edges that are outdated according to PrecedencePolicy
        std::vector< TaskVertexPtr > update_dependencies(TaskVertexPtr v)
        {
            std::unique_lock<std::shared_mutex> wrlock(v->out_edges_mutex);

            std::vector< TaskVertexPtr > revoked_vertices;

            for( unsigned i = 0; i < v->out_edges.size(); ++i)
            {
                TaskVertexPtr follower_vertex = v->out_edges[i].lock();

                if( !PrecedencePolicy::is_serial(v->template get_task<Task>(), follower_vertex->template get_task<Task>()) )
                {
                    revoked_vertices.push_back(follower_vertex);

                    std::remove_if(
                       std::begin(follower_vertex->in_edges),
                       std::end(follower_vertex->in_edges),
                       [v](std::weak_ptr<PrecedenceGraphVertex> prev) { return prev.lock() == v; });

                    v->out_edges.erase(std::next(std::begin(v->out_edges), i--));
                }
            }

            return revoked_vertices;
        }
    };

    /*!
     */
    struct TaskSpace : std::enable_shared_from_this<TaskSpace>
    {
        using TaskVertexPtr = std::shared_ptr<PrecedenceGraphVertex>;

        moodycamel::ConcurrentQueue<std::unique_ptr<ITask>> queue;
        std::shared_ptr<IPrecedenceGraph> precedence_graph;
        std::optional<std::weak_ptr<PrecedenceGraphVertex>> parent;

        TaskSpace(
            std::shared_ptr<IPrecedenceGraph> precedence_graph,
            std::optional<std::weak_ptr<PrecedenceGraphVertex>> parent = std::nullopt)
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
        std::optional<TaskVertexPtr> next()
        {
            // we need to lock the graph uniquely here to preserve the insertion order,
            // since `try_dequeue()` is lock-free
            std::unique_lock<std::shared_mutex> wrlock(precedence_graph->mutex);

            std::unique_ptr<ITask> task;
            if(queue.try_dequeue(task))
            {
                // insert the new task upfront, so we iterate from latest to earliest.
                auto it = precedence_graph->tasks.insert(
                    std::begin(precedence_graph->tasks),
                    std::make_shared<PrecedenceGraphVertex>(std::move(task), this->shared_from_this()));
                //wrlock.unlock();

                // from now on, the precedence graph can be shared-locked,
                // since out_edges has its separate mutex
                precedence_graph->init_dependencies(it);

                return *it;
            }
            else
                return std::nullopt;
        }

        //! add task to the queue, will be processed lazily by `next()`
        void push(std::unique_ptr<ITask>&& task)
        {
            queue.enqueue(std::move(task));
        }

        void remove(TaskVertexPtr v)
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
