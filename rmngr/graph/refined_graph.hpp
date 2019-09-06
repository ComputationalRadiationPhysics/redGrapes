
/*
 * @file rmngr/refined_graph.hpp
 */

#pragma once

#include <unordered_map>
#include <vector>
#include <memory> // std::unique_ptr<>

#include <akrzemi/optional.hpp>
#include <boost/graph/copy.hpp>
#include <rmngr/graph/util.hpp>

namespace rmngr
{

struct FlagInterface
{
    virtual void clear() = 0;
};

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
	    : uptodate(nullptr)
	    , parent(nullptr)
        {}

        RefinedGraph(RefinedGraph&& g)
	  : uptodate(g.uptodate)
	  , refinements(g.refinements)
	  , m_graph(g.m_graph)
	  , parent(g.parent)
        {}

        /// get graph object
        Graph & graph(void)
        {
            return this->m_graph;
        }

        /// build the complete graph with its refinemets in target_graph
        template <typename MutableGraph>
        void copy(MutableGraph & target_graph)
        {
            boost::copy_graph(this->graph(), target_graph);
            this->copy_refinements(target_graph);
        }

        RefinedGraph * /* should be std::optional<std::reference_wrapper<RefinedGraph>> */
        find_refinement(ID parent)
        {
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
            if ( auto d = graph_find_vertex(a, this->graph()) )
            {
                std::vector<ID> trace;
                trace.push_back(a);

                if( this->parent )
                    trace.push_back(this->parent);

                return trace;
            }

            for (auto & r : this->refinements)
            {
                if (std::experimental::optional<std::vector<ID>> trace = r.second->backtrace(a))
                {
                    if( this->parent )
                        (*trace).push_back(this->parent);

                    return *trace;
                }
            }

            return std::experimental::nullopt;
        }

        template <typename Refinement>
        Refinement *
        make_refinement(ID parent)
        {
            Refinement * ptr = new Refinement();
            ptr->uptodate = this->uptodate;
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
            if (this->refinements.count(a) == 0)
            {
                if ( auto v = graph_find_vertex(a, this->graph()) )
                {
                    boost::clear_vertex(*v, this->graph());
                    boost::remove_vertex(*v, this->graph());

                    this->deprecate();
                    return true;
                }
                else
                {
                    for(auto & r : this->refinements)
                    {
                        if (r.second->finish(a))
                        {
                            if (boost::num_vertices(r.second->graph()) == 0)
                                this->refinements.erase(r.first);

                            this->deprecate();
                            return true;
                        }
                    }
                }
            }

            return false;
        }

        FlagInterface * uptodate;
        void deprecate(void)
        {
            if( this->uptodate != nullptr )
                this->uptodate->clear();
        }

    public:
        ID parent;

    private:
        std::unordered_map<ID, std::unique_ptr<RefinedGraph>> refinements;
        Graph m_graph;

        template <typename MutableGraph>
        void copy(MutableGraph & target_graph, VertexID parent)
        {
            // copy all vertices and edges from refinement into the scheduling graph
            struct vertex_copier
            {
                Graph & src;
                Graph & dest;
                VertexID parent;

                void
                operator() (VertexID in, VertexID out)
                {
                    ID a = graph_get(in, src);
                    dest[out] = a;
                    boost::add_edge(parent, out, dest);
                }
            };
            boost::copy_graph(
                this->graph(),
                target_graph,
                boost::vertex_copy( vertex_copier
            {
                this->graph(),
                target_graph,
                parent} )
            );
            this->copy_refinements(target_graph);
        }

        template <typename MutableGraph>
        void copy_refinements(MutableGraph & target_graph)
        {
            for (auto & r : this->refinements)
            {
                if ( auto parent = graph_find_vertex(r.first, target_graph) )
                    r.second->copy(target_graph, *parent);
                else
                    r.second->copy(target_graph);
            }
        }

}; // class RefinedGraph

} // namespace rmngr

