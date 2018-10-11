
/**
 * @file rmngr/resource/resource_user.hpp
 */

#pragma once

#include <vector>
#include <rmngr/resource/resource.hpp>

namespace rmngr
{

class ResourceUser
{
  public:
    ResourceUser() {}

    ResourceUser( std::vector<ResourceAccess> const & access_list_ )
        : access_list( access_list_ )
    {
    }

    static bool
    is_serial( ResourceUser const & a, ResourceUser const & b )
    {
        for ( ResourceAccess const & ra : a.access_list )
            for ( ResourceAccess const & rb : b.access_list )
                if ( ResourceAccess::is_serial( ra, rb ) )
                    return true;
        return false;
    }

    bool
    is_superset_of( ResourceUser const & a ) const
    {
        for ( ResourceAccess const & ra : a.access_list )
        {
            bool found = false;
            for ( ResourceAccess const & r : this->access_list )
                if ( r.is_same_resource( ra ) && r.is_superset_of( ra ) )
                    found = true;

            if ( !found )
                // a introduced a new resource
                return false;
        }
        return true;
    }

    static bool
    is_superset( ResourceUser const & a, ResourceUser const & b )
    {
      return a.is_superset_of(b);
    }

    // protected:
    std::vector<ResourceAccess> access_list;
}; // class ResourceUser

} // namespace rmngr
