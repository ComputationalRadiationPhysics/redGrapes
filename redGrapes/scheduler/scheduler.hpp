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

template < typename SchedulingGraph >
struct SchedulerBase
{
    using TaskID = typename SchedulingGraph::TaskID;
    using TaskPtr = typename SchedulingGraph::TaskPtr;
    SchedulingGraph & graph;

    std::atomic_flag uptodate;

    SchedulerBase( SchedulingGraph & graph )
        : graph(graph)
    {
        uptodate.clear();
        graph.notify_hook = [this]{ uptodate.clear(); };
    }

    void notify()
    {
        this->graph.notify();
    }

    bool is_task_ready( TaskPtr & task )
    {
        return boost::in_degree( task.vertex, task.graph->graph() ) == 0;
    }
};

} // namespace redGrapes
