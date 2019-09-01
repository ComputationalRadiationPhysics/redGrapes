
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <akrzemi/optional.hpp>

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
    {}

    bool is_task_ready( Task * task )
    {
        if( auto task_id = graph_find_vertex( task, graph.precedence_graph.graph() ) )
            return boost::out_degree( *task_id, graph.precedence_graph.graph() ) == 0;
        else
            return true;
    }

    std::experimental::optional<Task*> find_task( std::function<bool(Task*)> pred )
    {
        for(
            auto it = boost::vertices(graph.precedence_graph.graph());
            it.first != it.second;
            ++ it.first
        )
        {
            auto task_id = *(it.first);
            auto task = graph_get( task_id, graph.precedence_graph.graph() );
            if( pred( task ) )
                return std::experimental::optional<Task*>(task);
        }

        return std::experimental::nullopt;
    }

    void remove_tasks( std::function<bool(Task*)> pred )
    {
        for(
            auto it = boost::vertices(graph.precedence_graph.graph());
            it.first != it.second;
            ++ it.first
        )
        {
            auto task = graph_get( *(it.first), graph.precedence_graph.graph() );
            if( pred( task ) )
                graph.precedence_graph.finish( task );
        }
    }
};

} // namespace rmngr

