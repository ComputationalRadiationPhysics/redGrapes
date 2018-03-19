
/**
 * @file rmngr/scheduling_graph.hpp
 */

#pragma once

#include <map>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/labeled_graph.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/property_map/function_property_map.hpp>

namespace rmngr
{

/** Implements all graph related operations for the scheduler
 *
 * @tparam ID type to identify nodes
 */
template<
    typename ID,
    typename ReadyMarker
    >
class SchedulingGraph
{
    private:
        using Graph = typename boost::adjacency_list<
                      boost::setS,
                      boost::vecS,
                      boost::bidirectionalS,
                      ID
                      >;

        using LabeledGraph = typename boost::labeled_graph<Graph, ID>;

    public:
        using VertexID = typename boost::graph_traits<Graph>::vertex_descriptor;
        using EdgeID = typename boost::graph_traits<Graph>::edge_descriptor;
        using VertexIterator = typename boost::graph_traits<Graph>::vertex_iterator;
        using InEdgeIterator = typename boost::graph_traits<Graph>::in_edge_iterator;

        /// return id from vertex_descriptor
        template <typename T_Graph>
        static inline ID graph_get(T_Graph& graph, VertexID v)
        {
            return boost::get(boost::vertex_bundle, graph)[v];
        }

        // TODO: factor out
        static inline std::pair<VertexID, bool> find_vertex(ID a, Graph& graph)
        {
            VertexIterator it, end;
            for( boost::tie(it, end) = boost::vertices(graph); it != end; ++it)
            {
                if( graph_get(graph, *it) == a )
                    return std::make_pair(*it, true);
            }

            return std::make_pair(VertexID{}, false);
        }

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
                    this->precedence_graph.add_vertex(a);
                    this->precedence_graph[a] = a;
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
                friend class SchedulingGraph;
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
            return (boost::out_degree(find_vertex(a, this->scheduling_graph).first, this->scheduling_graph) == 0);
        }

        /**
         * Recreate the scheduling-graph from refinements
         */
        void update_schedule(void)
        {
            // merge all refinements into one graph
            this->scheduling_graph.clear();

            for(auto const& r : this->refinements)
            {
                // copy all vertices and edges from refinement into the scheduling graph
                struct vertex_copier
                {
                    Graph& src;
                    Graph& dest;
                    VertexID parent;

                    void operator() (VertexID in, VertexID out)
                    {
                        ID a = graph_get(src, in);
                        dest[out] = a;
                        add_edge(parent, out, dest);
                    }
                };

                std::pair<VertexID, bool> parent = find_vertex( r.first, this->scheduling_graph );
                if( parent.second )
                {
                    // copy & add dependency to parent node
                    boost::copy_graph(r.second->precedence_graph.graph(),
                                      this->scheduling_graph,
                                      boost::vertex_copy( vertex_copier
                    {
                        r.second->precedence_graph.graph(),
                        this->scheduling_graph,
                        parent.first
                    }));
                }
                else
                {
                    // we have no parent node
                    boost::copy_graph(r.second->precedence_graph.graph(), this->scheduling_graph);
                }
            }

            // TODO: apply scheduling policy

            // check which vertices are ready
            VertexIterator it, end;
            for(boost::tie(it, end) = boost::vertices(this->scheduling_graph); it != end; ++it)
                this->update_ready(graph_get(scheduling_graph,*it));
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

        rmngr::observer_ptr< Refinement >
        make_refinement( ID parent )
        {
            this->refinements[ parent ] = std::unique_ptr<Refinement> ( new Refinement() );
            return rmngr::observer_ptr<Refinement>( this->refinements[parent] );
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

        Graph& graph(void)
        {
            return this->scheduling_graph;
        }

    private:
        /// list of sub-graphs
        std::map<ID, std::unique_ptr<Refinement>> refinements;

        /// main graph
        Graph scheduling_graph;
        ReadyMarker mark_ready;

        /// Mark ready if it is
        void update_ready(ID a)
        {
            if(this->is_ready(a))
                this->mark_ready(a);
        }
}; // class SchedulingGraph

} // namespace rmngr

