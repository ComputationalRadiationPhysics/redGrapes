
#pragma once

#include <rmngr/resource.hpp>
#include <rmngr/ioaccess.hpp>

namespace rmngr
{

template <typename Dependency=BoolDependency>
using IOResource = StaticResource<IOAccess, Dependency>;

}; // namespace rmngr

