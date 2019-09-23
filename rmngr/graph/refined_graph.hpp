
/*
 * @file rmngr/refined_graph.hpp
 */

#pragma once

#include <unordered_map>
#include <vector>
#include <memory> // std::unique_ptr<>
#include <mutex>

#include <akrzemi/optional.hpp>
#include <boost/graph/copy.hpp>
#include <rmngr/graph/util.hpp>

namespace rmngr
{

/**
 * Boost-Graph adaptor storing a tree of subgraphs
 * which refine a node.
 * Every vertex of a refinement has an edge to the
 * refinements root node.
 */
template <typename Graph>
class RefinedGraph
{
    private:
        using ID = typename Graph::vertex_property_type;
        using VertexID = typename boost::graph_traits<Graph>::vertex_descriptor;

    public:
        RefinedGraph()
        {}

        RefinedGraph(RefinedGraph&& g)
            : refinements(g.refinements)
            , m_graph(g.m_graph)
            , parent(g.parent)
        {}

        auto lock()
        {
            return std::unique_lock<std::recursive_mutex>( mutex );
        }

        /// get graph object
        Graph & graph(void)
        {
            return this->m_graph;
        }

    bool empty()
    {
        auto l = lock();
        return boost::num_vertices( graph() ) == 0;
    }

        RefinedGraph * /* should be std::optional<std::reference_wrapper<RefinedGraph>> */
        find_refinement(ID parent)
        {
            auto l = lock();
            auto it = this->refinements.find(parent);

            if (it != this->refinements.end())
                return it->second.get();
            else
            {
                for (auto & r : this->refinements)
                {
                    auto found = r.second->find_refinement(parent);
                    if (found)
                        return found;
                }

                return nullptr;
            }
        }

        RefinedGraph *
        find_refinement_containing(ID a)
        {
            auto l = lock();
            if ( auto d = graph_find_vertex(a, this->graph()) )
                return this;

            for (auto & r : this->refinements)
            {
                auto found = r.second->find_refinement_containing(a);
                if (found)
                    return found;
            }

            return nullptr;
        }

        std::experimental::optional<std::vector<ID>> backtrace(ID a)
        {
            auto l = lock();
            if ( auto d = graph_find_vertex(a, this->graph()) )
            {
                std::vector<ID> trace;
                trace.push_back(a);

                if( this->parent )
                    trace.push_back(*this->parent);

                return trace;
            }

            for (auto & r : this->refinements)
            {
                if (std::experimental::optional<std::vector<ID>> trace = r.second->backtrace(a))
                {
                    if( this->parent )
                        (*trace).push_back(*this->parent);

                    return *trace;
                }
            }

            return std::experimental::nullopt;
        }

        template <typename Refinement>
        Refinement *
        make_refinement(ID parent)
        {
            auto l = lock();
            Refinement * ptr = new Refinement( this );
            ptr->parent = parent;
            this->refinements[parent] = std::unique_ptr<RefinedGraph>(ptr);
            return ptr;
        }

        template <typename Refinement>
	Refinement *
        refinement(ID parent)
        {
            auto ref = this->find_refinement(parent);

            if (! ref)
            {
                auto base = this->find_refinement_containing(parent);
                if (base)
                    return base->template make_refinement<Refinement>(parent);

                // else: parent doesnt exist, return nullptr
            }

            return dynamic_cast<Refinement*>((RefinedGraph*)ref);
        }

        /// recursively remove a vertex
        /// does it belong here?
        virtual bool finish(ID a)
        {
            auto l = lock();
            if (this->refinements.count(a) == 0)
            {
                //std::cerr << "refined-graph: " << a << " hos no childs"<<std::endl;
                if ( auto v = graph_find_vertex(a, this->graph()) )
                {
                    boost::clear_vertex(*v, this->graph());
                    boost::remove_vertex(*v, this->graph());
                    mark_dirty();

                    //std::cerr << "removed VERTEX for task " << a << std::endl;

                    return true;
                }
                else
                {
                    for(auto & r : this->refinements)
                    {
                        if ( r.second->finish(a) )
                        {
                            if (boost::num_vertices(r.second->graph()) == 0)
                            {
                                this->refinements.erase(r.first);
                                this->mark_dirty();
                            }

                            return true;
                        }
                    }
                }
            }
            //else
            //    std::cerr << "refined-graph: "<<a << " has children" << std::endl;

            return false;
        }

    struct Iterator
    {
        RefinedGraph & r;
        typename boost::graph_traits< Graph >::vertex_iterator g_it;
        std::unique_ptr< std::pair< Iterator, Iterator > > sub;
        std::unique_lock< std::recursive_mutex > lock;

        ID operator* ()
        {
            if( !sub )
            {
                auto id = graph_get( *g_it, r.graph() );
                if( r.refinements.count(id) )
                    sub.reset( new std::pair<Iterator,Iterator>( r.refinements[id]->vertices() ) );
                else
                    return id;
            }

            if( sub->first == sub->second )
            {
                sub.reset(nullptr);
                return graph_get( *g_it, r.graph() );
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
                   Iterator{*this, g_it.first, nullptr, this->lock()},
                   Iterator{*this, g_it.second, nullptr}
               );
    }

    bool test_and_set()
    {
        bool u = uptodate.test_and_set();
        for( auto & r : refinements )
            u &= r.second->test_and_set();
        return u;
    }

    void mark_dirty()
    {
        this->uptodate.clear();
    }

    public:
        std::experimental::optional<ID> parent;

    private:
        std::atomic_flag uptodate;
        std::recursive_mutex mutex;
        std::unordered_map<ID, std::unique_ptr<RefinedGraph>> refinements;
        Graph m_graph;
}; // class RefinedGraph

} // namespace rmngr

