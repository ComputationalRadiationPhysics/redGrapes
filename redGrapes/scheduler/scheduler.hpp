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
#include <redGrapes/graph/scheduling_graph.hpp>
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
    std::atomic_bool finishing;

    SchedulerBase( std::shared_ptr<PrecedenceGraph> precedence_graph, size_t n_threads )
        : precedence_graph( precedence_graph )
        , scheduling_graph( n_threads )
        , finishing( false )
    {
        uptodate.clear();
    }

    void notify()
    {
        uptodate.clear();
        for( auto & thread : scheduling_graph.schedule )
            thread.notify();
    }

    void finish()
    {
        finishing = true;
        notify();
    }

    auto get_current_task()
    {
        return scheduling_graph.get_current_task();
    }

    void update_vertex( TaskPtr p )
    {
        scheduling_graph.update_vertex( p );
        notify();
    }

    void operator() ( std::function<bool()> const & pred = []{ return false; } )
    {
        auto l = thread::scope_level;
        while( !pred() && !( finishing && scheduling_graph.empty() ) )
            scheduling_graph.consume_job( pred );
        thread::scope_level = l;
    }
};

} // namespace redGrapes
