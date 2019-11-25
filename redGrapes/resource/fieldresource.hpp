/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/resource/fieldresource.hpp
 */

#pragma once

#include <limits>

#include <redGrapes/access/field.hpp>
#include <redGrapes/resource/resource.hpp>

namespace redGrapes
{

template <
    size_t dimension_t
>
struct FieldResource : Resource< access::FieldAccess<dimension_t> >
{
    using Resource<access::FieldAccess<dimension_t>>::make_access;
    ResourceAccess make_access(
        access::IOAccess io,
        std::array<access::AreaAccess, dimension_t> range
    ) const
    {
        return this->make_access(
                   access::FieldAccess<dimension_t>(
                       io,
                       access::ArrayAccess<
                           access::AreaAccess,
                           dimension_t
                       >(range)
                   )
               );
    }

#define OP(name)                                                                \
    inline ResourceAccess name (                                                \
        std::array<std::array<int,2>, dimension_t> const& range                 \
    ) const                                                                     \
    {                                                                           \
        return this->make_access(                                               \
                   access::IOAccess{access::IOAccess::name},                    \
                   *reinterpret_cast<                                           \
                       std::array<access::AreaAccess, dimension_t> const*       \
                    >(&range)                                                   \
               );                                                               \
    }                                                                           \
    inline ResourceAccess name (                                                \
        std::initializer_list< std::array<int,2> > init                         \
    ) const                                                                     \
    {                                                                           \
        std::array< std::array<int, 2>, dimension_t > range;                    \
        int i = 0;                                                              \
        for( auto r : init )                                                    \
            range[i++] = r;                                                     \
        return name ( range );                                                  \
    }                                                                           \
    inline ResourceAccess name (void) const                                     \
    {                                                                           \
        std::array< std::array<int, 2>, dimension_t > range;                    \
        for( auto & r : range )                                                 \
            r = {                                                               \
                std::numeric_limits<int>::min(),                                \
                std::numeric_limits<int>::max()                                 \
            };                                                                  \
        return name ( range );                                                  \
    }                                                                           \


    OP(read)
    OP(write)
    OP(aadd)
    OP(amul)

#undef OP
}; // struct FieldResource

}; // namespace redGrapes
