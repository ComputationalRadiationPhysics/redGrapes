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

} // namespace redGrapes

