/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/access/field.hpp
 */

#pragma once

#include <redGrapes/resource/access/io.hpp>
#include <redGrapes/resource/access/area.hpp>
#include <redGrapes/resource/access/combine.hpp>

namespace redGrapes
{
namespace access
{

template <
    size_t dimension_t
>
using FieldAccess = CombineAccess<
    IOAccess,
    ArrayAccess<
        AreaAccess,
        dimension_t,
        And_t
    >,
    And_t
>;

} // namespace access

} // namespace redGrapes
