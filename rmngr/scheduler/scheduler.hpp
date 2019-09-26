
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <akrzemi/optional.hpp>

#include <vector>

namespace rmngr
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
    {}

    bool is_task_ready( TaskID task )
    {
        auto r = graph.precedence_graph.find_refinement_containing( task );
        if( r )
        {
            auto l = r->lock();
            if( auto task_id = graph_find_vertex( task, r->graph() ) )
                return boost::in_degree( *task_id, graph.precedence_graph.graph() ) == 0;
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

} // namespace rmngr

