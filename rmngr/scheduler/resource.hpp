
#pragma once

#include <stdexcept>
#include <rmngr/resource/resource_user.hpp>
#include <rmngr/scheduler/scheduler.hpp>

namespace rmngr
{

struct ResourceUserPolicy : DefaultSchedulingPolicy
{
    using ProtoProperty = ResourceUser;

    void update_property(
        ProtoProperty& s,
        RuntimeProperty&,
        std::vector< ResourceAccess > const & access_list
    )
    {
        ResourceUser n( access_list );

        if( s.is_superset_of( n ) )
            s.access_list = access_list;
        else
            throw std::runtime_error("rmngr: updated access is no superset");
    }
};

} // namespace rmngr

