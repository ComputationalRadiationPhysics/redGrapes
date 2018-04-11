
/**
 * @file rmngr/resource_user.hpp
 */

#pragma once

#include <vector>
#include <rmngr/resource.hpp>

namespace rmngr
{

class ResourceUser
{
    public:
        ResourceUser(std::vector<ResourceAccess> const& access_list_)
          : access_list(access_list_)
        {}

        static bool
        is_serial(
            ResourceUser const & a,
            ResourceUser const & b
        )
        {
            for(ResourceAccess const& ra : a.access_list)
            {
                for(ResourceAccess const& rb : b.access_list)
                {
                    if(ResourceAccess::is_serial(ra, rb))
                        return true;
                }
            }
            return false;
        }

        //protected:
        std::vector<ResourceAccess> access_list;
}; // class ResourceUser

} // namespace rmngr

