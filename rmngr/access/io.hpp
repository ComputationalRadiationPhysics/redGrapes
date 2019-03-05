
/**
 * @file rmngr/access/ioaccess.hpp
 */

#pragma once

#include <boost/graph/adjacency_matrix.hpp>
#include <rmngr/access/dependency_manager.hpp>

namespace rmngr
{
namespace access
{

/**
 * Implements the concept @ref AccessPolicy
 */
struct IOAccess
{
    enum Mode
    {
        root,
        read,
        write,
        aadd,
        amul,
    } mode;

    IOAccess()
      : mode(root) {}

    IOAccess( enum Mode mode_ )
      : mode(mode_) {}

    static bool
    is_serial(
        IOAccess a,
        IOAccess b
    )
    {
        return m().is_serial(a.mode, b.mode);
    }

    bool
    is_superset_of(IOAccess a) const
    {
        return m().is_superset(this->mode, a.mode);
    }

  private:
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

    static StaticDependencyManager<Graph, Initializer, 5> const & m(void)
    {
        static StaticDependencyManager<
            Graph,
            Initializer,
            5
        > const m;
        return m;
    }
}; // struct IOAccess

} // namespace access

} // namespace rmngr

