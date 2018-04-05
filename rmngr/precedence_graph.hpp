
#pragma once

#include <list>
#include <unordered_map>
#include <memory> // std::unique_ptr<>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/labeled_graph.hpp>
#include <boost/graph/copy.hpp>

#include <rmngr/refined_graph.hpp>
#include <rmngr/observer_ptr.hpp>
#include <iostream>

namespace rmngr
{

struct AllSequential
{
    template <typename ID>
    bool
    is_sequential( ID, ID )
    {
        return true;
    }
};

struct AllParallel
{
    template <typename ID>
    bool
    is_sequential( ID, ID )
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
        }

        /// a precedes b
        void add_edge(ID a, ID b)
        {
            boost::add_edge(
                graph_find_vertex(b, this->graph()).first,
                graph_find_vertex(a, this->graph()).first,
                this->graph()
            );
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
    public PrecedenceGraph<Graph>,
    private EnqueuePolicy
{
    private:
        using ID = typename Graph::vertex_property_type;

    public:
        void push(ID a)
        {
            this->add_vertex(a);

            for(auto b : this->queue)
            {
                if( this->EnqueuePolicy::is_sequential(b, a) )
                    this->add_edge(b, a);
            }

            this->queue.insert(this->queue.begin(), a);
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

