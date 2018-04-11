
/**
 * @file rmngr/ioaccess.hpp
 */

#pragma once

#include <boost/graph/adjacency_matrix.hpp>

#include <rmngr/dependency_manager.hpp>
#include <rmngr/resource.hpp>

namespace rmngr
{

struct IOAccess
{
    enum
    {
        root,
        read,
        write,
        aadd,
        amul,
    } mode;

    static bool
    is_serial(
        IOAccess a,
        IOAccess b
    )
    {
        using Graph = boost::adjacency_matrix<boost::undirectedS>;
        struct Initializer
        {
            void operator() (Graph& g) const
            {
                // atomic operations
                boost::add_edge(root, read, g);
                boost::add_edge(root, aadd, g);
                boost::add_edge(root, amul, g);

                // non-atomic
                boost::add_edge(root, write, g);
                boost::add_edge(write, write, g);
            };
        }; // struct Initializer

        static StaticDependencyManager<
            Graph,
            Initializer,
            5
        > const m;
        return m.is_serial(a.mode, b.mode);
    }

}; // struct IOAccess

struct IOResource : public Resource< IOAccess >
{
#define OP(name) \
  inline ResourceAccess name (void) \
  { return this->make_access(IOAccess{IOAccess::name}); }

  OP(read)
  OP(write)
  OP(aadd)
  OP(amul)

#undef OP
}; // struct IOResource

} // namespace rmngr

