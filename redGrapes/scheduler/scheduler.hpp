/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <akrzemi/optional.hpp>
#include <redGrapes/graph/util.hpp>

#include <vector>

namespace redGrapes
{

template < typename TaskProperties, typename SchedulingGraph >
struct SchedulerBase
{
    TaskContainer< TaskProperties > & tasks;
    SchedulingGraph & graph;

    std::atomic_flag uptodate;

    using TaskID = typename TaskContainer< TaskProperties >::TaskID;

    SchedulerBase( TaskContainer< TaskProperties > & tasks, SchedulingGraph & graph )
        : tasks(tasks)
        , graph(graph)
    {
        uptodate.clear();
        graph.notify_hook = [this]{ uptodate.clear(); };
    }

    void notify()
    {
        this->graph.notify();
    }

    bool is_task_ready( TaskID task )
    {
        auto r = graph.precedence_graph.find_refinement_containing( task );
        if( r )
        {
            auto l = r->lock();
            if( auto task_id = graph_find_vertex( task, r->graph() ) )
                return boost::in_degree( *task_id, r->graph() ) == 0;
        }
        return false;
    }

    std::experimental::optional<TaskID> find_task( std::function<bool(TaskID)> pred = [](TaskID){ return true; } )
    {
        for(
            auto it = graph.precedence_graph.vertices();
            it.first != it.second;
            ++ it.first
        )
        {
            auto task = *(it.first);
            if( pred( task ) )
                return std::experimental::optional<TaskID>(task);
        }

        return std::experimental::nullopt;
    }

    std::vector<TaskID> collect_tasks( std::function<bool(TaskID)> pred = [](TaskID){ return true; } )
    {
        std::vector<TaskID> selection;
        for(
            auto it = graph.precedence_graph.vertices();
            it.first != it.second;
            ++ it.first
        )
        {
            auto task = *(it.first);
            if( pred( task ) )
                selection.push_back( task );
        }

        return selection;
    }
};

} // namespace redGrapes
