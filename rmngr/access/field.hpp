
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
        Or_t
    >,
    And_t
>;

} // namespace access

} // namespace rmngr

