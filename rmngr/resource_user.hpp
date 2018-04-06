#pragma once

#include <vector>
#include <memory>

#include <rmngr/resource.hpp>

namespace rmngr
{

class ResourceUser
{
    public:
        struct CheckDependency
        {
            static inline bool is_sequential(ResourceUser const& a, ResourceUser const& b)
            {
                for(auto ra : a.access_list)
                {
                    for(auto rb : b.access_list)
                    {
                        if(ra->check_dependency(*rb))
                            return true;
                    }
                }
                return false;
            }
        };

        ResourceUser(std::vector<std::shared_ptr<ResourceAccess>> const& access_list_)
            : access_list(access_list_)
        {}

        //protected:
        std::vector<std::shared_ptr<ResourceAccess>> access_list;
}; // class ResourceUser

} // namespace rmngr

