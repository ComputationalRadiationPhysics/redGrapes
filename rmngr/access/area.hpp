
/**
 * @file rmngr/access/area.hpp
 */

#pragma once

#include <array>
#include <iostream>

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

    friend std::ostream& operator<<(std::ostream& out, AreaAccess const& a)
    {
        out << "AreaAccess::[" << a[0] << ", " << a[1] << "]";
	return out;
    }
}; // struct AreaAccess

} // namespace access

} // namespace rmngr

