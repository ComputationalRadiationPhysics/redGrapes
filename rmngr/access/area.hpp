
/**
 * @file rmngr/access/area.hpp
 */

#pragma once

#include <array>

namespace rmngr
{
namespace access
{

struct AreaAccess : std::array<int, 2>
{
    AreaAccess()
      : std::array<int, 2>
        {
          std::numeric_limits<int>::min(),
          std::numeric_limits<int>::max()
        }
    {}

    AreaAccess(std::array<int, 2> a)
      : std::array<int, 2>(a) {}

    static bool
    is_serial(
        AreaAccess const & a,
        AreaAccess const & b
    )
    {
        return !(
            (a[1] < b[0]) ||
            (a[0] > b[1])
        );
    }
}; // struct AreaAccess

} // namespace access

} // namespace rmngr

