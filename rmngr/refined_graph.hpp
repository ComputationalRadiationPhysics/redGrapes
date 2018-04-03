
/*
 * @file rmngr/refined_graph.hpp
 */

#pragma once

#include <unordered_map>
#include <memory> // std::unique_ptr<>

#include <boost/graph/copy.hpp>

#include <rmngr/observer_ptr.hpp>
#include <rmngr/graph_util.hpp>

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

        observer_ptr<RefinedGraph>
        find_refinement(ID parent)
        {
            auto it = this->refinements.find(parent);

            if (it != this->refinements.end())
                return it->second;
            else
            {
                for (auto & r : this->refinements)
                {
                    observer_ptr<RefinedGraph> found = r.second->find_refinement(parent);
                    if (found)
                        return found;
                }

                return nullptr;
            }
        }

        template <typename Refinement>
        observer_ptr<Refinement>
        make_refinement(ID parent)
        {
            Refinement* ptr = new Refinement();
            this->refinements[parent] = std::unique_ptr<RefinedGraph>(ptr);
            return observer_ptr<Refinement>(ptr);
        }

        template <typename Refinement>
        observer_ptr<Refinement>
        refinement(ID parent)
        {
            observer_ptr<Refinement> ref((Refinement*)(void*)this->find_refinement(parent));

            if (! ref)
                ref = make_refinement<Refinement>(parent);

            return ref;
        }

        /// recursively remove a vertex
        /// does it belong here?
        virtual bool finish(ID a)
        {
            auto v = graph_find_vertex(a, this->graph());

            if (v.second)
            {
                boost::clear_vertex(v.first, this->graph());
                boost::remove_vertex(v.first, this->graph());
                return true;
            }
            else
            {
                for( auto & r : this->refinements )
                {
                    if ( r.second->finish(a) )
                        return true;
                }
            }

            return false;
        }

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
                std::pair<VertexID, bool> parent = graph_find_vertex(r.first, target_graph);

                if (parent.second)
                    r.second->copy(target_graph, parent.first);
                else
                    r.second->copy(target_graph);
            }
        }

}; // class RefinedGraph

} // namespace rmngr

