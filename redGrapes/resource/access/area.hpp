/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/access/area.hpp
 */

#pragma once

#include <limits>
#include <array>
#include <fmt/format.h>

namespace redGrapes
{
namespace access
{

struct AreaAccess : std::array<size_t, 2>
{
    AreaAccess()
    {
        ( *this )[ 0 ] = std::numeric_limits< size_t >::min();
        ( *this )[ 1 ] = std::numeric_limits< size_t >::max();
    }

    AreaAccess(std::array<size_t, 2> a)
      : std::array<size_t, 2>(a) {}

    static bool
    is_serial(
        AreaAccess const & a,
        AreaAccess const & b
    )
    {
        return !(
            (a[1] <= b[0]) ||
            (a[0] >= b[1])
        );
    }

    bool
    is_superset_of(AreaAccess const & a) const
    {
        return (
            ((*this)[0] <= a[0]) &&
            ((*this)[1] >= a[1])
        );
    }

    bool operator==(AreaAccess const & other) const
    {
        return (*this)[0] == other[0] && (*this)[1] == other[1];
    }

}; // struct AreaAccess



} // namespace access

} // namespace redGrapes

template <>
struct fmt::formatter< redGrapes::access::AreaAccess >
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::access::AreaAccess const & acc,
        FormatContext & ctx
    )
    {
        return format_to(
                   ctx.out(),
                   "{{ \"area\" : {{ \"begin\" : {}, \"end\" : {} }} }}",
                   acc[0],
                   acc[1]
               );
    }
};

