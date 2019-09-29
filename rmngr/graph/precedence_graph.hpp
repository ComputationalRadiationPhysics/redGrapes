
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
#include <boost/graph/reverse_graph.hpp>

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
        auto add_vertex(ID a)
        {
            auto l = this->lock();
            auto v = boost::add_vertex(a, this->graph());
            this->mark_dirty();

            return v;
        }

        /// a precedes b
        void add_edge(ID a, ID b)
        {
            auto l = this->lock();
            boost::add_edge(
                *graph_find_vertex(a, this->graph()),
                *graph_find_vertex(b, this->graph()),
                this->graph()
            );
            this->mark_dirty();
        }

        /// remove edges which don't satisfy the precedence policy
        std::vector<ID> remove_out_edges(ID id, std::function<bool(ID)> const & pred)
        {
            auto l = this->lock();
            auto v = *graph_find_vertex(id, this->graph());

            std::vector<ID> selection;
            std::vector<typename boost::graph_traits<Graph>::vertex_descriptor> vertices;

            for(auto it = boost::out_edges(v, this->graph()); it.first != it.second; ++it.first)
            {
                auto other_vertex = boost::target(*(it.first), this->graph());
                auto other_id = graph_get(other_vertex, this->graph());
                if( pred( other_id ) )
                {
                    selection.push_back( other_id );
                    vertices.push_back( other_vertex );
                }
            }

            for( auto other_vertex : vertices )
                boost::remove_edge(v, other_vertex, this->graph());

            this->mark_dirty();

            return selection;
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
        using VertexID = typename boost::graph_traits<Graph>::vertex_descriptor;
        std::experimental::optional<EnqueuePolicy> policy;

        bool is_serial( ID a, ID b )
        {
            if( policy )
                return policy->is_serial( a, b );
            return true;
        }

    void assert_superset( ID super, ID sub )
    {
        if( policy )
            policy->assert_superset( super, sub );
    }

    public:
        QueuedPrecedenceGraph()
            : policy( std::experimental::nullopt )
        {}

        QueuedPrecedenceGraph( EnqueuePolicy const & policy )
            : policy( policy )
        {}

    template < typename T_Graph >
    QueuedPrecedenceGraph( RefinedGraph<T_Graph> * p )
    {
        auto parent = dynamic_cast<QueuedPrecedenceGraph<T_Graph, EnqueuePolicy>*>(p);
        if( parent )
            this->policy = parent->policy;
        else
            this->policy = std::experimental::nullopt;
    }

        VertexID push(ID a)
        {
            auto l = this->lock();
            if( this->parent )
                this->assert_superset( *this->parent, a );

            VertexID v = this->add_vertex(a);

            struct Visitor : boost::default_dfs_visitor
            {
                Graph const & g;
                std::unordered_set<ID>& discovered;

                Visitor(Graph const & g, std::unordered_set<ID>& d)
                    : g(g)
                    , discovered(d)
                {}

                void discover_vertex(VertexID v, boost::reverse_graph<Graph> const&)
                {
                    this->discovered.insert(graph_get(v, g));
                }
            };

            std::unordered_set<ID> indirect_dependencies;
            Visitor vis(this->graph(), indirect_dependencies);

            std::unordered_map<VertexID, boost::default_color_type> vertex2color;
            auto colormap = boost::make_assoc_property_map(vertex2color);

            VertexID i = *graph_find_vertex(a, this->graph());
            for(auto b : this->queue)
            {
                if( this->is_serial(b, a) && indirect_dependencies.count(b) == 0 )
                {
                    this->add_edge(b, a);
                    boost::depth_first_visit(boost::make_reverse_graph(this->graph()), i, vis, colormap);
                }
            }

            this->queue.insert(this->queue.begin(), a);

            return v;
        }

        auto update_vertex(ID a)
        {
            return this->remove_out_edges( a, [this,a](ID b){ return !this->is_serial(a, b); } );
	}

        bool finish(ID a)
        {
            auto l = this->lock();
            if( this->PrecedenceGraph<Graph>::finish(a) )
            {
                auto it = std::find(this->queue.begin(), this->queue.end(), a);
                if (it != this->queue.end())
                    this->queue.erase(it);

                return true;
            }
            else
                return false;
        }

    private:
        std::list<ID> queue;
}; // class QueuedPrecedenceGraph

} // namespace rmngr

