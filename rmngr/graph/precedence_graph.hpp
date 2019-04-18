
#pragma once

#include <list>
#include <unordered_map>
#include <unordered_set>
#include <memory> // std::unique_ptr<>
#include <stdexcept> // std::runtime_error

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/labeled_graph.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/copy.hpp>

#include <rmngr/graph/refined_graph.hpp>
#include <iostream>

namespace rmngr
{

struct AllSequential
{
    template <typename ID>
    static bool
    is_serial( ID, ID )
    {
        return true;
    }
};

struct AllParallel
{
    template <typename ID>
    static bool
    is_serial( ID, ID )
    {
        return false;
    }
};

/**
 * Base class
 */
template <typename Graph>
class PrecedenceGraph : public RefinedGraph<Graph>
{
    private:
        using ID = typename Graph::vertex_property_type;

    public:
        void add_vertex(ID a)
        {
            boost::add_vertex(a, this->graph());
            this->deprecate();
        }

        /// a precedes b
        void add_edge(ID a, ID b)
        {
            boost::add_edge(
                graph_find_vertex(b, this->graph()).first,
                graph_find_vertex(a, this->graph()).first,
                this->graph()
            );
            this->deprecate();
        }

        /// remove edges which don't satisfy the precedence policy
        template <typename PrecedencePolicy>
        void update_vertex(ID id)
        {
            struct Predicate
            {
                ID id;
                Graph const & graph;

                bool operator() ( typename boost::graph_traits<Graph>::edge_descriptor edge ) const
                {
                    auto src = boost::source(edge, graph);
                    return ( ! PrecedencePolicy::is_serial( id, graph_get(src, graph) ) );
                }
            };

            Predicate pred{id, this->graph()};
            auto v = graph_find_vertex(id, this->graph()).first;
            boost::remove_in_edge_if(v, pred, this->graph());

            this->deprecate();
        }
}; // class PrecedenceGraph

/**
 * Precedence-graph generated from a queue
 * using an enqueue-policy
 */
template<
    typename Graph,
    typename EnqueuePolicy
>
class QueuedPrecedenceGraph :
    public PrecedenceGraph<Graph>
{
    private:
        using ID = typename Graph::vertex_property_type;

    public:
        void push(ID a)
        {
            if( this->parent )
	        EnqueuePolicy::assert_superset( this->parent, a );

            this->add_vertex(a);

            using VertexID = typename boost::graph_traits<Graph>::vertex_descriptor;
            struct Visitor : boost::default_dfs_visitor
            {
                std::unordered_set<ID>& discovered;

                Visitor(std::unordered_set<ID>& d)
                    : discovered(d)
                {}

                void discover_vertex(VertexID v, Graph const& g)
                {
                    this->discovered.insert(graph_get(v, g));
                }
            };

            std::unordered_set<ID> indirect_dependencies;
            Visitor vis(indirect_dependencies);

            std::unordered_map<VertexID, boost::default_color_type> vertex2color;
            auto colormap = boost::make_assoc_property_map(vertex2color);

            VertexID i = graph_find_vertex(a, this->graph()).first;
            for(auto b : this->queue)
            {
                if( EnqueuePolicy::is_serial(b, a) && indirect_dependencies.count(b) == 0 )
                {
                    this->add_edge(b, a);
                    boost::depth_first_visit(this->graph(), i, vis, colormap);
                }
            }

            this->queue.insert(this->queue.begin(), a);
        }

        void update_vertex(ID a)
        {
            this->PrecedenceGraph<Graph>::template update_vertex< EnqueuePolicy >( a );
	}

        bool finish(ID a)
        {
            auto it = std::find(this->queue.begin(), this->queue.end(), a);

            if (it != this->queue.end())
                this->queue.erase(it);

            return this->PrecedenceGraph<Graph>::finish(a);
        }

    private:
        std::list<ID> queue;
}; // class QueuedPrecedenceGraph

} // namespace rmngr

