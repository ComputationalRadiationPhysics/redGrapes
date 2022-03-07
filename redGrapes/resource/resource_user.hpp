/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/resource/resource_user.hpp
 */

#pragma once

#include <list>
#include <redGrapes/resource/resource.hpp>

#include <fmt/format.h>

namespace redGrapes
{

class ResourceUser
{
  public:
    ResourceUser()
        : scope_level( dispatch::thread::scope_level ) {}

    ResourceUser( std::list<ResourceAccess> const & access_list_ )
        : access_list( access_list_ )
        , scope_level( dispatch::thread::scope_level )
    {}

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

            if ( !found && ra.scope_level() <= scope_level )
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

    unsigned int scope_level;
    std::list<ResourceAccess> access_list;
}; // class ResourceUser

} // namespace redGrapes

template <>
struct fmt::formatter<
    redGrapes::ResourceUser
>
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::ResourceUser const & r,
        FormatContext & ctx
    )
    {
        auto out = ctx.out();
        out = fmt::format_to( out, "[" );

        for( auto it = r.access_list.begin(); it != r.access_list.end(); )
        {
            out = fmt::format_to( out, "{}", *it );
            if( ++it != r.access_list.end() )
                out = fmt::format_to( out, "," );
        }

        out = fmt::format_to( out, "]" );
        return out;
    }
};

