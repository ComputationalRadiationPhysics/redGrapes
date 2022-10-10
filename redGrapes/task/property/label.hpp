/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/property/label.hpp
 */

#pragma once

#include <fmt/format.h>

namespace redGrapes
{

struct LabelProperty
{
    std::string label;

    template < typename TaskBuilder >
    struct Builder
    {
        TaskBuilder & builder;

        Builder( TaskBuilder & builder )
            : builder(builder)
        {}

        TaskBuilder & label( std::string const & l )
        {
            builder.task->label = l;
            return builder;
        }
    };

    struct Patch
    {
        template <typename PatchBuilder>
        struct Builder
        {
            Builder( PatchBuilder & ) {}
        };
    };

    void apply_patch( Patch const & ) {}
};

} // namespace redGrapes

template <>
struct fmt::formatter< redGrapes::LabelProperty >
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::LabelProperty const & label_prop,
        FormatContext & ctx
    )
    {
        return format_to(
                   ctx.out(),
                   "\"label\" : \"{}\"",
                   label_prop.label
               );
    }
};

