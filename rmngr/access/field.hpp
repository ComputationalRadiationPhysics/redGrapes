/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file rmngr/access/field.hpp
 */

#pragma once

#include <rmngr/access/io.hpp>
#include <rmngr/access/area.hpp>
#include <rmngr/access/combine.hpp>

namespace rmngr
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

} // namespace rmngr done
