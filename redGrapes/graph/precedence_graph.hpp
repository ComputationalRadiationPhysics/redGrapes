/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

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

#include <redGrapes/graph/recursive_graph.hpp>
#include <iostream>

#include <spdlog/spdlog.h>

namespace redGrapes
{

/*! EnqueuePolicy where all vertices are connected
 */
struct AllSequential
{
    template <typename ID>
    static bool
    is_serial( ID, ID )
    {
        return true;
    }
};

/*! EnqueuePolicy where no vertex has edges
 */
struct AllParallel
{
    template <typename ID>
    static bool
    is_serial( ID, ID )
    {
        return false;
    }
};

/*!
 * Base class to manage a graph
 */
template <
    typename T,
    template < class > typename Graph = DefaultGraph
>
class PrecedenceGraph : public RecursiveGraph< T, Graph >
{
public:
    using typename RecursiveGraph< T, Graph >::VertexID;

    virtual ~PrecedenceGraph() {}

    /*! add a new vertex
     *
     * @return
     */
    virtual VertexID push( T a ) = 0;

    /*! Update all outgoing edges of a vertex
     *
     * @return std::vector of all following vertices whose edge has been removed
     */
    virtual std::vector< VertexID > update_vertex( VertexID v ) = 0;

    /*! remove vertex
     */
    virtual void finish( VertexID v ) = 0;

    //! create child graph of same type
    virtual PrecedenceGraph * default_child(
        std::weak_ptr< RecursiveGraph< T, Graph > > parent_graph,
        VertexID parent_vertex
    ) = 0;

protected:
    //! remove edges from vertex to its followers, where pred does not satisfy the follower
    auto remove_out_edges(
        VertexID vertex,
        std::function< bool (T const&) > const & pred
    )
    {
        std::vector<VertexID> vertices;

        for(
            auto it = boost::out_edges( vertex, this->graph() );
            it.first != it.second;
            ++it.first
        )
        {
            auto other_vertex = boost::target( *(it.first), this->graph() );
            auto & other = graph_get( other_vertex, this->graph() );
            if( pred( other.first ) )
                vertices.push_back( other_vertex );
        }

        for( auto other_vertex : vertices )
            boost::remove_edge( vertex, other_vertex, this->graph() );

        return vertices;
    }
}; // class PrecedenceGraph

/*! Specialized precedence-graph that is constructed from a queue and an EnqueuePolicy.
 *
 * Vertices are added in a specific order (via the push() method).
 * On insertion, the new vertex is compared against all previously
 * inserted vertices. The EnqueuePolicy then decides, to which previously
 * created vertex an edge is created.
 */
template<
    typename T,
    typename EnqueuePolicy,
    template < class > typename Graph = DefaultGraph
>
class QueuedPrecedenceGraph
    : public PrecedenceGraph< T, Graph >
{
    public:
        using VertexID = typename PrecedenceGraph< T, Graph >::VertexID;

        QueuedPrecedenceGraph()
        {}

        QueuedPrecedenceGraph(
            std::weak_ptr< RecursiveGraph< T, Graph > > parent_graph,
            VertexID parent_vertex
        )
        {
            this->parent_graph = parent_graph;
            this->parent_vertex = parent_vertex;
        }

        PrecedenceGraph< T, Graph > * default_child(
            std::weak_ptr< RecursiveGraph< T, Graph > > parent_graph,
            VertexID parent_vertex
        )
        {
            return new QueuedPrecedenceGraph( parent_graph, parent_vertex );
        }
    
        /*! Add vertex to the graph according to the EnqueuePolicy
         */
        VertexID push( T a )
        {
            if( auto graph = this->parent_graph.lock() )
            {
                auto parent_lock = graph->shared_lock();
                EnqueuePolicy::assert_superset( graph_get( this->parent_vertex, graph->graph() ).first, a );
            }

            VertexID v = boost::add_vertex( std::make_pair(a, std::shared_ptr<RecursiveGraph<T,Graph>>(nullptr)), this->graph() );

            struct Visitor : boost::default_dfs_visitor
            {
                using G = Graph<std::pair<T,std::shared_ptr<RecursiveGraph<T,Graph>>>>;
                G const & g;
                std::unordered_set<VertexID>& discovered;

                Visitor(G const & g, std::unordered_set<VertexID>& d)
                    : g(g)
                    , discovered(d)
                {}

                void discover_vertex(VertexID v, boost::reverse_graph<G> const&)
                {
                    this->discovered.insert(v);
                }
            };

            std::unordered_set<VertexID> indirect_dependencies;
            Visitor vis(this->graph(), indirect_dependencies);

            std::unordered_map<VertexID, boost::default_color_type> vertex2color;
            auto colormap = boost::make_assoc_property_map(vertex2color);

            for( auto b : this->queue )
            {
                T const & prop = graph_get(b, this->graph()).first;
                if( EnqueuePolicy::is_serial(prop, a) && indirect_dependencies.count(b) == 0 )
                {
                    boost::add_edge(b, v, this->graph());
                    boost::depth_first_visit(boost::make_reverse_graph(this->graph()), b, vis, colormap);
                }
            }

            this->queue.insert(this->queue.begin(), v);

            return v;
        }

        /*! Update all outgoing edges of a vertex
         * @return std::vector of all following vertices whose edge has been removed
         */
        std::vector< VertexID > update_vertex( VertexID a )
        {
            return this->remove_out_edges(a, [this,a](T const & b){ return !EnqueuePolicy::is_serial(graph_get(a, this->graph()).first, b); } );
	}

        /*! Remove vertex from graph including all its edges
         */
        void finish( VertexID vertex )
        {
            boost::clear_vertex( vertex, this->graph() );
            boost::remove_vertex( vertex, this->graph() );

            auto it = std::find( this->queue.begin(), this->queue.end(), vertex );
            if ( it != this->queue.end() )
                this->queue.erase( it );
            else
            {
                spdlog::error("QueuedPrecedenceGraph: removed vertex {} which is not in queue", vertex);
                throw std::runtime_error("Queuedprecedencegraph: removed element not in queue");
            }
        }

private:
    std::list< VertexID > queue;
}; // class QueuedPrecedenceGraph

} // namespace redGrapes
