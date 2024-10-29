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

#include <fmt/format.h>

#include <array>
#include <limits>

namespace redGrapes
{
    namespace access
    {
        // Must be in increasing order
        struct RangeAccess : std::array<size_t, 2>
        {
            RangeAccess()
            {
                (*this)[0] = std::numeric_limits<size_t>::min();
                (*this)[1] = std::numeric_limits<size_t>::max();
            }

            RangeAccess(std::array<size_t, 2> a) : std::array<size_t, 2>(a)
            {
            }

            bool is_synchronizing() const
            {
                return (*this)[0] == std::numeric_limits<size_t>::min()
                       && (*this)[1] == std::numeric_limits<size_t>::max();
            }

            static bool is_serial(RangeAccess const& a, RangeAccess const& b)
            {
                return !((a[1] <= b[0]) || (a[0] >= b[1]));
            }

            bool is_superset_of(RangeAccess const& a) const
            {
                return (((*this)[0] <= a[0]) && ((*this)[1] >= a[1]));
            }

            bool operator==(RangeAccess const& other) const
            {
                return (*this)[0] == other[0] && (*this)[1] == other[1];
            }

        }; // struct AreaAccess


    } // namespace access

} // namespace redGrapes

template<>
struct fmt::formatter<redGrapes::access::RangeAccess>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::access::RangeAccess const& acc, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(), "{{ \"area\" : {{ \"begin\" : {}, \"end\" : {} }} }}", acc[0], acc[1]);
    }
};
