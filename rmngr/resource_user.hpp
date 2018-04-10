
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

        bool is_serial(ResourceUser const& r) const
        {
            for(auto const& ra : this->access_list)
            {
                for(auto const& rb : r.access_list)
                {
                    if(ra.is_serial(rb))
                        return true;
                }
            }
            return false;
        }

        //protected:
        std::vector<ResourceAccess> access_list;
}; // class ResourceUser

} // namespace rmngr

