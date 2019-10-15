/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file rmngr/resource/resource_user.hpp
 */

#pragma once

#include <list>
#include <rmngr/resource/resource.hpp>

namespace rmngr
{

class ResourceUser
{
  public:
    ResourceUser() {}

    ResourceUser( std::list<ResourceAccess> const & access_list_ )
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
                if ( r.is_superset_of( ra ) )
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

    std::list<ResourceAccess> access_list;

    friend std::ostream& operator<<(std::ostream& out, ResourceUser const& r)
    {
        out << "ResourceUser {" << std::endl;
        for( auto & access : r.access_list )
            out << access << "," << std::endl; 
	out << "}";
	return out;
    }
}; // class ResourceUser

} // namespace rmngr
