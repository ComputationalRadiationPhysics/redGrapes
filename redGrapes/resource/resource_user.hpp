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
#include <redGrapes/context.hpp>

#include <fmt/format.h>

namespace redGrapes
{

struct ResourceEntry
{
    std::shared_ptr< ResourceBase > resource;
    int task_idx; // index in task list of resource
};

class ResourceUser
{
  public:    
    ResourceUser()
        : scope_level( scope_depth() )
        , access_list()
    {
        access_list.reserve(1024);
    }

    ResourceUser( std::vector<ResourceAccess> const & access_list_ )
        : access_list( access_list_ )
        , scope_level( scope_depth() )
    {
        build_unique_resource_list();
    }

    void add_resource_access( ResourceAccess ra )
    {
        this->access_list.push_back(ra);
        std::shared_ptr<ResourceBase> r = ra.get_resource();
        if( std::find_if(unique_resources.begin(), unique_resources.end(), [r](ResourceEntry const & e){ return e.resource == r; }) == unique_resources.end() )
            unique_resources.push_back(ResourceEntry{ r, -1 });
    }

    void rm_resource_access( ResourceAccess ra )
    {
        auto it = std::find(this->access_list.begin(), this->access_list.end(), ra);
        this->access_list.erase(it);
    }

    void build_unique_resource_list()
    {
        unique_resources.clear();
        unique_resources.reserve(access_list.size());
        for( auto & ra : access_list )
        {
            std::shared_ptr<ResourceBase> r = ra.get_resource();
            if( std::find_if(unique_resources.begin(), unique_resources.end(), [r](ResourceEntry const & e){ return e.resource == r; }) == unique_resources.end() )
            unique_resources.push_back(ResourceEntry{ r, -1 });            
        }
    }

    bool has_sync_access( std::shared_ptr< ResourceBase > res )
    {
        for( auto & ra : access_list )
        {
            if(
               ra.get_resource() == res &&
               ra.is_synchronizing()
            )
                return true;
        }

        return false;
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

    std::vector<ResourceAccess> access_list;
    std::vector<ResourceEntry> unique_resources;
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

