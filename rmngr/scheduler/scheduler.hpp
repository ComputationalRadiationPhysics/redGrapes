
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <akrzemi/optional.hpp>

#include <vector>

namespace rmngr
{

template < typename SchedulingGraph >
struct SchedulerBase
{
    SchedulingGraph & graph;

    using Task = typename SchedulingGraph::Task;
    using TaskID = typename boost::graph_traits< typename SchedulingGraph::P_Graph >::vertex_descriptor;

    SchedulerBase( SchedulingGraph & graph )
        : graph(graph)
    {
    }

    bool is_task_ready( Task * task )
    {
        auto r = graph.precedence_graph.find_refinement_containing( task );
        if( r )
            if( auto task_id = graph_find_vertex( task, r->graph() ) )
                return boost::in_degree( *task_id, graph.precedence_graph.graph() ) == 0;
        return false;
    }

    std::experimental::optional<Task*> find_task( std::function<bool(Task*)> pred )
    {
        for(
            auto it = graph.precedence_graph.vertices();
            it.first != it.second;
            ++ it.first
        )
        {
            auto task = *(it.first);
            if( pred( task ) )
                return std::experimental::optional<Task*>(task);
        }

        return std::experimental::nullopt;
    }

    std::vector<Task*> collect_tasks( std::function<bool(Task*)> pred )
    {
        std::vector<Task*> selection;
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

} // namespace rmngr

