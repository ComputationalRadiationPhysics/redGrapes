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

#include <fmt/format.h>
#include <fmt/color.h>

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

template <>
struct fmt::formatter<
    redGrapes::access::IOAccess
>
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::access::IOAccess const & acc,
        FormatContext & ctx
    )
    {
        std::string mode_str;

        switch(acc.mode)
	{
        case redGrapes::access::IOAccess::root: mode_str = "root"; break;
        case redGrapes::access::IOAccess::read: mode_str = "read"; break;
        case redGrapes::access::IOAccess::write: mode_str = "write"; break;
        case redGrapes::access::IOAccess::aadd: mode_str = "atomicAdd"; break;
        case redGrapes::access::IOAccess::amul: mode_str = "atomicMul"; break;
	}

        return fmt::format_to(
                   ctx.out(),
                   "{{ \"IOAccess\" : \"{}\" }}",
                   mode_str
               );
    }
};

