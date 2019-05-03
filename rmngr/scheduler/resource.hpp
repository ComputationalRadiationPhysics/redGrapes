
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

template <typename T>
struct ResourceEnqueuePolicy
{
    static bool is_serial(T const & a, T const & b)
    {
        return rmngr::ResourceUser::is_serial(
                   a->template proto_property< rmngr::ResourceUserPolicy >(),
		   b->template proto_property< rmngr::ResourceUserPolicy >());
    }
    static void assert_superset(T const & super, T const & sub)
    {
        auto r_super = super->template proto_property< rmngr::ResourceUserPolicy >();
        auto r_sub = sub->template proto_property< rmngr::ResourceUserPolicy >();
        if(! rmngr::ResourceUser::is_superset( r_super, r_sub ))
        {
            std::stringstream stream;
            stream << "Not allowed: " << std::endl
		   << r_super << std::endl
		   << "is no superset of " << std::endl
	           << r_sub << std::endl;
            throw std::runtime_error(stream.str());
        }
    }
};

} // namespace rmngr

