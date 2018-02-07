
#pragma once

#include <unordered_set>
#include <map>
#include <vector>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/labeled_graph.hpp>

#include <rmngr/queue.hpp>

namespace rmngr
{

template <typename T, typename ReadyMarker, typename DependencyCheck>
class DependencyGraph : public Queue<T, ReadyMarker>
{
    protected:
        using typename Queue<T, ReadyMarker>::ID;
        using Graph = typename boost::adjacency_list<boost::listS, boost::listS, boost::bidirectionalS, ID>;
        using LabeledGraph = typename boost::labeled_graph<Graph, ID>;
        using VertexID = typename boost::graph_traits<Graph>::vertex_descriptor;
        using InEdgeIterator = typename boost::graph_traits<Graph>::in_edge_iterator;

        LabeledGraph dependency_graph;

    public:
        DependencyGraph(ReadyMarker const& mark_ready_)
            : Queue<T, ReadyMarker>(mark_ready_)
        {}

        bool is_ready(ID id)
        {
            return (boost::out_degree(this->dependency_graph.vertex(id), this->dependency_graph) == 0);
        }

    private:
        void push_(ID id)
        {
            this->dependency_graph.add_vertex(id);
            this->dependency_graph[id] = id;

            struct Visitor : boost::default_dfs_visitor
            {
                std::unordered_set<VertexID>& discovered;

                Visitor(std::unordered_set<VertexID>& d)
                    : discovered(d)
                {}

                void discover_vertex(VertexID v, Graph const& g)
                {
                    this->discovered.insert(v);
                }
            };

            std::unordered_set<VertexID> indirect_dependencies;
            Visitor vis(indirect_dependencies);

            std::map<VertexID, boost::default_color_type> vertex2color;
            auto colormap = boost::make_assoc_property_map(vertex2color);

            // reverse time and only add new dependencies
            for(ID i : this->queue)
            {
                if(indirect_dependencies.count(this->dependency_graph.vertex(i)) == 0)
                {
                    if(DependencyCheck::check((*this)[id], (*this)[i]))
                    {
                        boost::add_edge_by_label(id, i, this->dependency_graph);
                        boost::depth_first_visit(this->dependency_graph.graph(), this->dependency_graph.vertex(i), vis, colormap);
                    }
                }
            }
        }

        void finish_(ID id)
        {
            // generate list of potentially ready vertices
            std::vector<ID> next_vertices;

            InEdgeIterator ei, ei_end;
            for(boost::tie(ei, ei_end) = boost::in_edges(this->dependency_graph.vertex(id), this->dependency_graph); ei != ei_end; ++ei)
            {
                VertexID v = boost::source(*ei, this->dependency_graph);
                ID id = boost::get(boost::vertex_bundle, this->dependency_graph)[v];
                next_vertices.push_back(id);
            }

            // remove vertex & edges
            boost::clear_vertex_by_label(id, this->dependency_graph);
            boost::remove_vertex(id, this->dependency_graph);

            // check for new ready nodes
            for(ID v : next_vertices)
                this->update_ready(v);
        }
}; // class DependencyGraph

}; // namespace rmngr

