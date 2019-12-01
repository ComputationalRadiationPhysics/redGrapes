/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/*
 * @file redGrapes/recursive_graph.hpp
 */

#pragma once

#include <limits>
#include <vector>
#include <memory> // std::unique_ptr<>
#include <shared_mutex>

#include <akrzemi/optional.hpp>
#include <redGrapes/graph/util.hpp>

namespace std
{
using shared_mutex = shared_timed_mutex;
}

namespace redGrapes
{

template <typename T>
using DefaultGraph =
    boost::adjacency_list<
        boost::setS,
        boost::listS,
        boost::bidirectionalS,
        T
    >;

/**
 * Boost-Graph adaptor storing a tree of subgraphs
 * which refine a node.
 * Every vertex of a refinement has an edge to the
 * refinements root node.
 */
template <
    typename T,
    template <class> typename T_Graph = DefaultGraph
>
class RecursiveGraph
{
public:
    using Graph = T_Graph< std::pair<T, std::shared_ptr<RecursiveGraph>> >;
    using VertexID = typename boost::graph_traits<Graph>::vertex_descriptor;

    virtual ~RecursiveGraph() {}

    auto shared_lock()
    {
        return std::shared_lock<std::shared_mutex>( mutex );
    }

    auto unique_lock()
    {
        return std::unique_lock<std::shared_mutex>( mutex );
    }

    /// get graph object
    Graph & graph(void)
    {
        return this->m_graph;
    }

    bool empty()
    {
        auto l = shared_lock();
        return boost::num_vertices( m_graph ) == 0;
    }

    void add_subgraph(VertexID vertex, std::shared_ptr<RecursiveGraph> subgraph)
    {
        auto l = unique_lock();
        graph_get( vertex, m_graph ).second = subgraph;
    }

    void remove_vertex(VertexID vertex)
    {
        auto l = unique_lock();
        boost::clear_vertex(vertex, m_graph);
        boost::remove_vertex(vertex, m_graph);
    }

    struct Iterator
    {
        RecursiveGraph & r;
        typename boost::graph_traits< Graph >::vertex_iterator g_it;
        std::unique_ptr< std::pair< Iterator, Iterator > > sub;
        std::shared_lock< std::shared_mutex > lock;

        T const & operator* ()
        {
            if( !sub )
            {
                auto child = graph_get( *g_it, r.graph() );
                if( child.second )
                    sub.reset( new std::pair<Iterator,Iterator>( child.second->vertices() ) );
                else
                    return child.first;
            }

            if( sub->first == sub->second )
            {
                sub.reset(nullptr);
                return graph_get( *g_it, r.graph() ).first;
            }
            else
                return *(sub->first);
        }

        bool operator== ( Iterator const & other )
        {
            return g_it == other.g_it;
        }

        bool operator!= ( Iterator const & other )
        {
            return g_it != other.g_it;
        }

        void operator++ ()
        {
            if( sub )
                ++(sub->first);
            else
                ++g_it;
        }
    };

    std::pair<Iterator, Iterator> vertices()
    {
        auto g_it = boost::vertices( graph() );
        return std::make_pair(
                   Iterator{*this, g_it.first, nullptr, this->shared_lock()},
                   Iterator{*this, g_it.second, nullptr}
               );
    }

    template <typename Result>
    void collect_vertices(
        std::vector<Result> & collection,
        std::function<std::experimental::optional<Result>(T const &)> const & filter_map,
        size_t limit = std::numeric_limits<size_t>::max()
    )
    {
        auto l = shared_lock();
        for(auto it = boost::vertices(m_graph); it.first != it.second && collection.size() < limit; ++it.first)
        {
            auto & w = graph_get(*it.first, m_graph);
            if( auto element = filter_map(w.first) )
                collection.push_back(*element);

            if( w.second )
                w.second->collect_vertices(collection, filter_map, limit);
        }
    }

    // Graphviz
    void write_dot(
        std::ostream & out,
        std::function<unsigned int(T const&)> const & id,
        std::function<std::string(T const&)> const & label,
        std::function<std::string(T const&)> const & color
    )
    {
        out << "digraph G {" << std::endl
            << "compound = true;" << std::endl
            << "graph [fontsize=10 fontname=\"Verdana\"];" << std::endl
            << "node [shape=record fontsize=10 fontname=\"Verdana\"];" << std::endl;

        this->write_refinement_dot( out, id, label, color );

        out << "}" << std::endl;
    }

    void write_refinement_dot(
        std::ostream & out,
        std::function<unsigned int(T const&)> const & id,
        std::function<std::string(T const&)> const & label,
        std::function<std::string(T const&)> const & color
    )
    {
        auto l = shared_lock();
        for( auto it = boost::vertices(graph()); it.first != it.second; ++it.first )
        {
            auto v = graph_get(*(it.first), graph());
            if( v.second )
            {
                out << "subgraph cluster_" << id(v.first) << " {" << std::endl
                    << "node [style = filled];" << std::endl
                    << "label = \"" << label(v.first) << "\";" << std::endl
                    << "color = " << color(v.first) << ";" << std::endl;

                v.second->write_refinement_dot( out, id, label, color );

                out << "root_" << id(v.first) << " [style=invis];" << std::endl;

                out << "};" << std::endl;
            }
            else
                out << id(v.first) << " [label = \"" << label(v.first) << "\", color = " << color(v.first) << "];" << std::endl;
        }

        for( auto it = boost::edges(graph()); it.first != it.second; ++it.first )
        {
            auto a = graph_get(boost::source( *(it.first), graph() ), graph() );
            auto b = graph_get(boost::target( *(it.first), graph() ), graph() );

            if( a.second )
            {
                out << "root_" << id(a.first) << " -> " << id(b.first) << " [ltail = cluster_" << id(a.first) << "];" << std::endl;
            }
            else
                out << id(a.first) << " -> " << id(b.first) << ";" << std::endl;
        }
    }

protected:
    VertexID parent_vertex;
    std::weak_ptr<RecursiveGraph> parent_graph;

    Graph m_graph;
    std::shared_mutex mutex;
}; // class RefinedGraph

} // namespace redGrapes
