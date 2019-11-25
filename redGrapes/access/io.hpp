/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/access/ioaccess.hpp
 */

#pragma once

#include <boost/graph/adjacency_matrix.hpp>
#include <redGrapes/access/dependency_manager.hpp>
#include <iostream>

namespace redGrapes
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

    bool operator==(IOAccess const & other) const
    {
        return this->mode == other.mode;
    }

    bool
    is_superset_of(IOAccess a) const
    {
        return m().is_superset(this->mode, a.mode);
    }

    friend std::ostream& operator<<(std::ostream& out, IOAccess const& a)
    {
        out << "IOAccess::";
	switch(a.mode)
	{
            case root: out << "Root"; break;
            case read: out << "Read"; break;
            case write: out << "Write"; break;
            case aadd: out << "AtomicAdd"; break;
            case amul: out << "AtomicMul"; break;
	}
	return out;
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

} // namespace redGrapes
