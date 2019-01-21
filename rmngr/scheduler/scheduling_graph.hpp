
/**
 * @file rmngr/graph/scheduling_graph.hpp
 */

#pragma once

#include <map>
#include <algorithm>
#include <atomic>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/property_map/function_property_map.hpp>

#include <rmngr/graph/util.hpp>
#include <rmngr/graph/refined_graph.hpp>
#include <rmngr/graph/precedence_graph.hpp>

namespace rmngr
{

/** Implements all graph related operations for the scheduler
 *
 * @tparam ID type to identify nodes
 */
template <
    typename Graph,
    typename RefinementGraph = Graph
>
class SchedulingGraph
{
    public:
        using ID = typename Graph::vertex_property_type;
        using VertexID = typename boost::graph_traits<Graph>::vertex_descriptor;
        using EdgeID = typename boost::graph_traits<Graph>::edge_descriptor;

        SchedulingGraph(
            std::atomic_flag volatile * uptodate_,
            observer_ptr<RefinedGraph<RefinementGraph>> main_ref
        )
            : uptodate(uptodate_), main_refinement(main_ref)
        {
            this->main_refinement->uptodate = this->uptodate;
        }

        bool empty(void) const
        {
            return (boost::num_vertices(this->scheduling_graph) == 0);
        }

        /** Check if a node has no dependencies
         *
         * @param a node to check
         * @return true if node is ready
         */
        bool is_ready(ID a)
        {
            // if node has no out edges, it does not depend on others
            return (boost::out_degree(
                        graph_find_vertex(a, this->scheduling_graph).first,
                        this->scheduling_graph
                    ) == 0);
        }

        /**
         * Recreate the scheduling-graph from refinements
         */
        void update(void)
        {
            // merge all refinements into one graph
            this->scheduling_graph.clear();
            this->main_refinement->copy(this->scheduling_graph);
        }

        /** Remove a node from the graphs and reschedule
         *
         * @param a id of node
         * @return if finish complete or if it has to
         *         wait until finishing of refinements
         */
        bool finish(ID a)
        {
            bool finished = this->main_refinement->finish(a);
            if( finished )
                this->uptodate->clear();

            return finished;
        }

        /** Write the current scheduling-graph as graphviz
         *
         * @param out output stream
         * @param names_ptr property map for node names
         * @param colors_ptr property map for node colors
         * @param label undertitle of graph
         */
        template <typename NamePropertyMap, typename ColorPropertyMap>
        void write_graphviz(
            std::ostream & out,
            NamePropertyMap names_ptr,
            ColorPropertyMap colors_ptr,
            std::string label = "Scheduling Graph"
        )
        {
            auto ids = boost::make_function_property_map<VertexID>(
                [this](VertexID const & id)
                {
                    return size_t(id);
                }
            );
            auto names = boost::make_function_property_map<VertexID>(
                [this, &names_ptr](VertexID const & id)
                {
                    return names_ptr[graph_get(id, this->scheduling_graph)];
                }
            );
            auto colors = boost::make_function_property_map<VertexID>(
                [this, &colors_ptr](VertexID const & id)
                {
                    return colors_ptr[graph_get(id,this->scheduling_graph)];
                }
            );

            boost::dynamic_properties dp;
            dp.property("id", ids);
            dp.property("label", names);
            dp.property("fillcolor", colors);
            dp.property("label", boost::make_constant_property<Graph *>(label));
            dp.property("rankdir", boost::make_constant_property<Graph *>
                (std::string("RL")));
            dp.property("shape", boost::make_constant_property<VertexID>
                (std::string("box")));
            dp.property("style", boost::make_constant_property<VertexID>
                (std::string("rounded,filled")));
            dp.property("dir", boost::make_constant_property<EdgeID>
                (std::string("back")));

            boost::write_graphviz_dp(out, this->scheduling_graph, dp,
                std::string("id"));
        }

        Graph & graph(void)
        {
            return this->scheduling_graph;
        }

    private:
        std::atomic_flag volatile * uptodate;
        observer_ptr<RefinedGraph<RefinementGraph>> main_refinement;
        Graph scheduling_graph;
}; // class SchedulingGraph

} // namespace rmngr

