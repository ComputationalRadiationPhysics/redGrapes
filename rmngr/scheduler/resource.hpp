
#pragma once

#include <rmngr/resource/resource_user.hpp>
#include <rmngr/scheduler/scheduler.hpp>

namespace rmngr
{

struct ResourceUserPolicy : rmngr::DefaultSchedulingPolicy
{
    using ProtoProperty = rmngr::ResourceUser;
};

} // namespace rmngr

