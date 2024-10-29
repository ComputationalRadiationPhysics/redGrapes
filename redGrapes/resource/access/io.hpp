/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/access/ioaccess.hpp
 */

#pragma once

#include <fmt/format.h>

#include <cstdint>

namespace redGrapes
{
    namespace access
    {
        /**
         * Implements the concept @ref AccessPolicy
         */
        struct IOAccess
        {
            enum Mode : uint8_t
            {
                write,
                read,
                aadd,
                amul,
            } mode;

            IOAccess() : mode(write)
            {
            }

            IOAccess(enum Mode mode_) : mode(mode_)
            {
            }

            bool is_synchronizing() const
            {
                return mode == write;
            }

            bool operator==(IOAccess const& other) const
            {
                return this->mode == other.mode;
            }

            static bool is_serial(IOAccess a, IOAccess b)
            {
                return !(
                    (a.mode == read && b.mode == read) || (a.mode == aadd && b.mode == aadd)
                    || (a.mode == amul && b.mode == amul));
            }

            bool is_superset_of(IOAccess a) const
            {
                return (this->mode == write) || (this->mode == a.mode);
            }
        }; // struct IOAccess

    } // namespace access

} // namespace redGrapes

template<>
struct fmt::formatter<redGrapes::access::IOAccess>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::access::IOAccess const& acc, FormatContext& ctx) const
    {
        std::string mode_str;

        switch(acc.mode)
        {
        case redGrapes::access::IOAccess::read:
            mode_str = "read";
            break;
        case redGrapes::access::IOAccess::write:
            mode_str = "write";
            break;
        case redGrapes::access::IOAccess::aadd:
            mode_str = "atomicAdd";
            break;
        case redGrapes::access::IOAccess::amul:
            mode_str = "atomicMul";
            break;
        }

        return fmt::format_to(ctx.out(), "{{ \"IOAccess\" : \"{}\" }}", mode_str);
    }
};
