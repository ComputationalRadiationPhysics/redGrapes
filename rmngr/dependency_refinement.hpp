
#pragma once

#include <unordered_set>
#include <map>
#include <vector>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>

#include <rmngr/queue.hpp>

namespace rmngr
{

template <typename ID, typename DependencyCheck>
class DependencyRefinement : public Queue<ID>, public SchedulingGraph<ID>::Refinement
{
        using VertexID = boost::graph_traits<SchedulingGraph<ID>::Graph>::vertex_descriptor;

    public:
        void push(ID a)
        {
            this->add_vertex(a);

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
                    if(DependencyCheck::check(a, i))
                    {
                        this->add_edge(a, i);
                        boost::depth_first_visit(this->graph(), this->vertex(i), vis, colormap);
                    }
                }
            }

            this->Queue<ID>::push(a);
        }

        void finish(ID a)
        {
            this->Refinement<ID>::finish(a);
            this->Queue<ID>::finish(a);
        }
}; // class DependencyRefinement

}; // namespace rmngr

