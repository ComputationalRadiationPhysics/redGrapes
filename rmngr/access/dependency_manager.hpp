
/**
 * @file rmngr/dependency_manager.hpp
 */

#pragma once

#include <stack>
#include <boost/graph/graph_traits.hpp>

#include <iostream>

namespace rmngr
{

/**
 * Manages a Graph to determine if two vertices are independent or
 * if they need to be serial
 *
 * Vertex B is serial to A if B can be reached from a without going backwards.
 *
 * TODO: more meaningful name ?
 */
template <typename Graph>
class DependencyManager
{
    using VertexID = typename boost::graph_traits<Graph>::vertex_descriptor;

    public:
        DependencyManager(size_t n)
          : m_graph(n) {}

        /**
         * adds edges to make implicit dependencies explicit
         */
        void update(void)
        {
            // TODO: use boost-graph algorithm ?

            auto vs = boost::vertices(this->graph());
            for(auto vi = vs.first; vi != vs.second; ++vi)
            {
                VertexID v = *vi;

                std::vector<bool> visited( boost::num_vertices(this->graph()) );
                std::stack<std::pair<VertexID, VertexID> > stack;

                stack.push(std::make_pair(v, v));

                while(! stack.empty())
                {
                    // get parent vertex
                    VertexID current, parent;
                    boost::tie(current, parent) = stack.top();

                    stack.pop();

                    auto es = boost::out_edges(current, this->graph());
                    for(auto ei = es.first; ei != es.second; ++ei)
                    {
                        VertexID w = boost::target(*ei, this->graph());
                        if(! visited[w])
                        {
                            boost::add_edge(v, w, this->graph());
                            if(w != parent)
                                stack.push(std::make_pair(w, current));
                        }
                    }

                    visited[current] = true;
                }
            }
        }

        bool is_serial(VertexID a, VertexID b) const
        {
            return boost::edge(a, b, this->graph()).second;
        }

        void print(void) const
        {
            auto vs = boost::vertices( this->graph() );

            for( auto x = vs.first; x != vs.second; ++x)
            {
                for( auto y = vs.first; y != vs.second; ++y)
                {
                    std::cout << is_serial(*x, *y) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        Graph& graph(void)
        {
            return this->m_graph;
        }

        Graph const& graph(void) const
        {
            return this->m_graph;
        }

    private:
        Graph m_graph;
}; // class DependencyManager

/** Can be used as static object to store immutable graphs
 *
 * Since we have no real compile-time code execution in C++,
 * we initialize the matrix at first instance.
 *
 * @tparam Graph graph model e.g. boost::adjacency_matrix
 * @tparam Initializer functor which adds the edges to the graph
 * @tparam N number of vertices
 */
template <
    typename Graph,
    typename Initializer,
    size_t N
>
class StaticDependencyManager : public DependencyManager<Graph>
{
    public:
        StaticDependencyManager()
          : DependencyManager<Graph>(N)
        {
            if( ! init )
            {
                Initializer initializer;
                initializer( this->graph() );
                this->update();
                init = true;
            }
        }

    private:
        static bool init;
}; // class StaticDependencyManager

template<typename Graph, typename Init, size_t N>
bool StaticDependencyManager<Graph, Init, N>::init = false;

} // namespace rmngr

