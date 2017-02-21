#pragma once

#include <cassert>
#include <stack>
#include <utility>
#include <unordered_map>
#include <boost/foreach.hpp>
#include <boost/graph/directed_graph.hpp>

#include <rmngr/matrix.hpp>

namespace rmngr
{

class BoolDependency
{
    public:
        BoolDependency()
            : state(true)
        {
        }

        BoolDependency(bool s)
            : state(s)
        {
        }

        operator bool() const
        {
            return this->state;
        }

        void operator&= (BoolDependency d)
        {
            this->state &= d.state;
        }

    private:
        bool state;
}; // class BoolDependency

template <typename Element, typename Dependency=BoolDependency>
class DependencyManager
{
    public:
        struct VertexProp
        {
            int matrix_id;
        };

        struct EdgeProp
        {
            Dependency dependency;
        };

        typedef boost::directed_graph<VertexProp, EdgeProp> DependencyGraph;
        typedef typename boost::graph_traits<DependencyGraph>::vertex_descriptor VertexID;
        typedef typename boost::graph_traits<DependencyGraph>::edge_descriptor EdgeID;
        typedef typename boost::graph_traits<DependencyGraph>::vertex_iterator VertexIterator;
        typedef typename boost::graph_traits<DependencyGraph>::in_edge_iterator InEdgeIterator;
        typedef typename boost::graph_traits<DependencyGraph>::out_edge_iterator OutEdgeIterator;

        // row depends on column
        class DependencyMatrix : public Matrix<Dependency>
        {
            public:
                DependencyMatrix()
                {
                }

                void update(DependencyGraph const& graph)
                {
                    int n = boost::num_vertices(graph);
                    this->resize(n,n);

                    // check dependencies of every vertex
                    VertexIterator vi, vi_end;
                    for(boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi)
                    {
                        VertexID v = *vi;
                        int i = graph[v].matrix_id;

                        std::vector<bool> visited(n);
                        std::vector<bool> keep(n);
                        std::deque<std::pair<VertexID, VertexID> > to_visit;
                        std::stack<std::pair<VertexID, VertexID> > stack(to_visit);

                        stack.push(std::make_pair(v, v));

                        while(! stack.empty())
                        {
                            // get parent vertex
                            VertexID current, parent;
                            boost::tie(current, parent) = stack.top();
                            int k = graph[current].matrix_id;

                            stack.pop();

                            OutEdgeIterator ei, ei_end;
                            for(boost::tie(ei, ei_end) = boost::out_edges(current, graph); ei != ei_end; ++ei)
                            {
                                VertexID w = boost::target(*ei, graph);
                                int j = graph[w].matrix_id;

                                if(! visited[j])
                                {
                                    // combine dependency
                                    (*this)(i,j) &= graph[*ei].dependency;
                                    keep[j] = true;

                                    if(w != parent)
                                        stack.push(std::make_pair(w, current));
                                }
                            }

                            visited[k] = true;
                        }

                        // delete all independencies
                        for(int j = 0; j < n; ++j)
                        {
                            if(! keep[j])
                                (*this)(i,j) &= false;
                        }
                    }

                    this->print();
                }
        };

        DependencyManager()
        {
            this->matrix_old = true;
        }

        VertexID add_vertex(Element const& elem)
        {
            int n = boost::num_vertices(this->dependency_graph);
            VertexID v = this->dependency_graph.add_vertex({n});
            this->elements[elem] = v;
            return v;
        }

        // a depends on b ; b influences a
        void add_dependency(VertexID const& a, VertexID const& b, Dependency dep=Dependency())
        {
            this->dependency_graph.add_edge(b, a, {dep});

            // this action deprecates the matrix
            this->matrix_old = true;
        }

        void add_dependency(Element const& a, Element const& b, Dependency dep=Dependency())
        {
            VertexID const ia = this->elements[a];
            VertexID const ib = this->elements[b];
            this->add_dependency(ia, ib, dep);
        }

        void update_matrix(void)
        {
            if(this->matrix_old)
            {
                this->dependency_matrix.update(this->dependency_graph);
                this->matrix_old = false;
            }
        }

        // check if a has to be executed after b
        Dependency check_dependency(VertexID const& a, VertexID const& b)
        {
            this->update_matrix();

            int matrix_id_a = this->dependency_graph[a].matrix_id;
            int matrix_id_b = this->dependency_graph[b].matrix_id;

            return this->dependency_matrix(matrix_id_a, matrix_id_b);
        }

        Dependency check_dependency(Element const& a, Element const& b)
        {
            return this->check_dependency(this->elements[a], this->elements[b]);
        }

    private:
        DependencyGraph dependency_graph;
        DependencyMatrix dependency_matrix;
        bool matrix_old;

        std::unordered_map<Element, VertexID> elements;
}; // class DependencyManager

} // namespace rmngr

