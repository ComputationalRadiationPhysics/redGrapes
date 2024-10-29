/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/access/field.hpp
 */

#pragma once

#include "redGrapes/resource/access/combine.hpp"
#include "redGrapes/resource/access/io.hpp"
#include "redGrapes/resource/access/range.hpp"

namespace redGrapes
{
    namespace access
    {

        template<size_t dimension_t>
        using FieldAccess = CombineAccess<IOAccess, ArrayAccess<RangeAccess, dimension_t, And_t>, And_t>;

    } // namespace access

} // namespace redGrapes
