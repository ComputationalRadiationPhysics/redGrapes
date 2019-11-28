/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <unordered_map>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <akrzemi/optional.hpp>
#include <redGrapes/graph/util.hpp>

#include <vector>

namespace redGrapes
{

template <
    typename TaskID,
    typename TaskPtr,
    typename PrecedenceGraph
>
struct SchedulerBase
{
    std::shared_ptr<PrecedenceGraph> precedence_graph;
    SchedulingGraph<TaskID, TaskPtr> scheduling_graph;

    std::atomic_flag uptodate;

    SchedulerBase( std::shared_ptr<PrecedenceGraph> precedence_graph, size_t n_threads )
        : precedence_graph( precedence_graph )
        , scheduling_graph( n_threads )
    {
        uptodate.clear();
        scheduling_graph.notify_hook = [this]{ uptodate.clear(); };
    }

    void notify()
    {
        this->scheduling_graph.notify();
    }

    void finish()
    {
        scheduling_graph.finish();
    }

    auto get_current_task()
    {
        return scheduling_graph.get_current_task();
    }

    void update_vertex( TaskPtr p )
    {
        scheduling_graph.update_vertex( p );
    }

    void operator() ( std::function<bool()> const & pred = []{ return false; } )
    {
        auto l = thread::scope_level;
        while( !pred() && !scheduling_graph.empty() )
            scheduling_graph.consume_job( pred );
        thread::scope_level = l;
    }

protected:
    bool is_task_ready( TaskPtr & task )
    {
        return boost::in_degree( task.vertex, task.graph->graph() ) == 0;
    }
};

} // namespace redGrapes
