#pragma once

#include <iostream>
#include <vector>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/graphviz.hpp>

namespace rmngr
{

struct DefaultCheck
{
    template <typename T>
    static inline bool check(T const& a, T const& b)
    {
        return true;
    }
};

struct DefaultLabel
{
    template <typename T>
    static inline std::string getLabel(T const& a)
    {
        return std::string("unnamed");
    }
};

template <typename Element, typename DependencyCheck=DefaultCheck, typename Label=DefaultLabel>
class Queue
{
    private:
        struct VertexProp
        {
            Element const& elem;
        };
        typedef boost::directed_graph<VertexProp> DependencyGraph;
        typedef typename boost::graph_traits<DependencyGraph>::vertex_descriptor VertexID;
        typedef typename boost::graph_traits<DependencyGraph>::edge_descriptor EdgeID;
        typedef typename boost::graph_traits<DependencyGraph>::in_edge_iterator InEdgeIterator;
        typedef typename boost::graph_traits<DependencyGraph>::out_edge_iterator OutEdgeIterator;

        DependencyGraph dependency_graph;
        std::vector<VertexID> pending;

        class label_writer
        {
            public:
                label_writer(DependencyGraph const& graph_)
                    : graph(graph_)
                {
                }

                template <class VertexOrEdge>
                void operator()(std::ostream& out, VertexOrEdge const& v) const
                {
                    out << "[label=\"" << Label::getLabel(this->graph[v].elem) << "\"]";
                }

            private:
                DependencyGraph const& graph;
        };

    public:
        Queue()
        {
        }

        ~Queue()
        {
        }

        void push(Element const& elem)
        {
            VertexID v = this->dependency_graph.add_vertex({elem});

            std::vector<VertexID> deplist;
            for(VertexID i : boost::adaptors::reverse(this->pending))
            {
                Element const& prev = this->dependency_graph[i].elem;
                if(DependencyCheck::check(elem, prev))
                {
                    // check if it is already dependent
                    for(VertexID j : boost::adaptors::reverse(deplist))
                    {
                        // iterate over parents (in vertices)
                        InEdgeIterator ei, ei_end;
                        for(boost::tie(ei, ei_end) = boost::in_edges(j, this->dependency_graph); ei != ei_end; ++ei)
                        {
                            if(boost::source(*ei, this->dependency_graph) == i) // v depends indirectly on i
                                goto push;
                        }
                    }

                    this->dependency_graph.add_edge(i, v);
push:
                    deplist.push_back(i);
                }
            }

            this->pending.push_back(v);
        }

        void write_dependency_graph(std::ostream& out)
        {
            boost::write_graphviz(out, this->dependency_graph, label_writer(this->dependency_graph));
        }
};

} // namespace rmngr

