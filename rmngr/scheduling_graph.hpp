
/**
 * @file rmngr/scheduling_graph.hpp
 */

#pragma once

#include <map>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/property_map/function_property_map.hpp>

namespace rmngr
{

/** Implements all graph related operations for the scheduler
 *
 * @tparam ID type to identify nodes
 */
template<typename ID, typename ReadyMarker>
class SchedulingGraph
{
    private:
        using Graph = typename boost::adjacency_list<boost::listS,
              boost::listS,
              boost::bidirectionalS,
              ID>;
        using LabeledGraph = typename boost::labeled_graph<Graph, ID>;
        using VertexID = typename boost::graph_traits<Graph>::vertex_descriptor;
        using EdgeID = typename boost::graph_traits<Graph>::edge_descriptor;
        using VertexIterator = typename boost::graph_traits<Graph>::vertex_iterator;
        using InEdgeIterator = typename boost::graph_traits<Graph>::in_edge_iterator;

        /// return id from vertex_descriptor
        static inline ID graph_get(LabeledGraph& graph, VertexID v)
        {
            return boost::get(boost::vertex_bundle, graph)[v];
        }

    public:
        SchedulingGraph(ReadyMarker const& mr)
            : mark_ready(mr) {}

        /**
         * Base class for handling the precedence-graphs, which get
         * composed to the scheduling graph
         */
        class Refinement
        {
            public:
                virtual ~Refinement() {};

                void add_vertex(ID a)
                {
                    this->dependency_graph.add_vertex(a);
                    this->dependency_graph[a] = a;
                }

                /// get vertex descriptor
                VertexID vertex(ID a)
                {
                    return this->dependency_graph.vertex(a);
                }

                /// a precedes b
                void add_edge(ID a, ID b)
                {
                    boost::add_edge_by_label(b, a, this->precedence_graph);
                }

                /// get graph object
                Graph& graph(void)
                {
                    return this->precedence_graph.graph();
                }

                virtual void finish(ID a)
                {
                    boost::clear_vertex_by_label(a, this->precedence_graph);
                    boost::remove_vertex(a, this->precedence_graph);
                };

            private:
                LabeledGraph precedence_graph;
        }; // class Refinement

        /** Check if a node has no dependencies
         *
         * @param a node to check
         * @return true if node is ready
         */
        bool is_ready(ID a)
        {
            // if node has no out edges, it does not depend on others
            return (boost::out_degree(this->scheduling_graph.vertex(a), this->scheduling_graph) == 0);
        }

        /**
         * Recreate the scheduling-graph from refinements
         */
        void update_schedule(void)
        {
            std::cout << "reschedule" << std::endl;

            // merge all refinements into one graph
            this->scheduling_graph.clear();

            for(auto const& r : this->refinements)
            {
                // copy all vertices and edges from refinement into the scheduling graph
                struct labeled_vertex_copier
                {
                    LabeledGraph& src;
                    LabeledGraph& dest;

                    void operator() (VertexID in, VertexID out) const
                    {
                        ID a = graph_get(src, in);
                        boost::put(boost::vertex_bundle, a, dest.graph());
                        dest[a] = a;
                    }
                };

                boost::copy_graph(r.second->precendence_graph,
                                  this->scheduling_graph,
                                  boost::vertex_index_map(boost::get(boost::vertex_bundle, r.second->precedence_graph)),
                                  boost::vertex_copy(labeled_vertex_copier{}));

                // wait for the root node
                VertexIterator it, end;
                for(boost::tie(it, end) = boost::vertices(r.second->precendence_graph); it != end; ++it)
                    boost::add_edge_by_label(r.first, graph_get(r.second->precedence_graph, *it));
            }

            // TODO

            // check which vertices are ready
            VertexIterator it, end;
            for(boost::tie(it, end) = boost::vertices(this->scheduling_graph); it != end; ++it)
                this->update_ready(graph_get(this->scheduling_graph, *it));
        }

        /** Remove a node from the graphs and reschedule
         *
         * @param a id of node
         */
        void finish(ID a)
        {
            if(this->refinements.count(a) == 0)
            {
                for(auto const& r : this->refinements)
                    r.second->finish(a);
            }

            this->update_schedule();
        }

        /** Write the current scheduling-graph as graphviz
         *
         * @param out output stream
         * @param names_ptr property map for node names
         * @param colors_ptr property map for node colors
         * @param label undertitle of graph
         */
        template <typename NamePropertyMap, typename ColorPropertyMap>
        void write_graphviz(std::ostream& out, NamePropertyMap names_ptr, ColorPropertyMap colors_ptr, std::string label="Scheduling Graph")
        {
            auto ids = boost::make_function_property_map<VertexID>([this](VertexID const& id)
            {
                return size_t((void*) graph_get(this->scheduling_graph, id));
            });
            auto names = boost::make_function_property_map<VertexID>([this, &names_ptr](VertexID const& id)
            {
                return names_ptr[graph_get(this->scheduling_graph, id)];
            });
            auto colors = boost::make_function_property_map<VertexID>([this, &colors_ptr](VertexID const& id)
            {
                return colors_ptr[graph_get(this->scheduling_graph, id)];
            });

            boost::dynamic_properties dp;
            dp.property("id", ids);
            dp.property("label", names);
            dp.property("fillcolor", colors);
            dp.property("label", boost::make_constant_property<LabeledGraph*>(label));
            dp.property("rankdir", boost::make_constant_property<LabeledGraph*>(std::string("RL")));
            dp.property("shape", boost::make_constant_property<VertexID>(std::string("box")));
            dp.property("style", boost::make_constant_property<VertexID>(std::string("rounded,filled")));
            dp.property("dir", boost::make_constant_property<EdgeID>(std::string("back")));

            boost::write_graphviz_dp(out, this->scheduling_graph, dp, std::string("id"));
        }

    private:
        /// list of sub-graphs
        std::map<ID, std::unique_ptr<Refinement>> refinements;

        /// main graph
        LabeledGraph scheduling_graph;
        ReadyMarker mark_ready;

        /// Mark ready if it is
        void update_ready(ID a)
        {
            if(this->is_ready(a))
                this->mark_ready(a);
        }
}; // class SchedulingGraph

} // namespace rmngr

