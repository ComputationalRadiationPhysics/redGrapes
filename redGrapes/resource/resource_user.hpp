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
#include <fmt/format.h>

#include <redGrapes/resource/resource.hpp>
#include <redGrapes/context.hpp>
#include <redGrapes/util/chunked_list.hpp>
#include <redGrapes/util/trace.hpp>

namespace redGrapes
{

struct ResourceEntry
{
    std::shared_ptr< ResourceBase > resource;
    ChunkedList< Task* >::MutBackwardIterator task_entry;

    bool operator==( ResourceEntry const & other ) const
    {
        return resource == other.resource;
    }
};

class ResourceUser
{
  public:    
    ResourceUser()
        : scope_level( scope_depth() )
        , access_list( 16 )
        , unique_resources( 16 )
    {
    }

    ResourceUser( ResourceUser const& other )
        : scope_level( other.scope_level )
        , access_list( memory::Allocator<uint8_t>(), other.access_list )
        , unique_resources( memory::Allocator<uint8_t>(), other.unique_resources )
    {
    }

    ResourceUser( std::initializer_list< ResourceAccess > list )
        : scope_level( scope_depth() )
    {
        for( auto & ra : list )
            add_resource_access(ra);
    }

    inline void add_resource_access( ResourceAccess ra )
    {
        this->access_list.push(ra);
        std::shared_ptr<ResourceBase> r = ra.get_resource();
        //unique_resources.erase(ResourceEntry{ r, -1 });
        unique_resources.push(ResourceEntry{ r, r->users.rend() });
    }

    void rm_resource_access( ResourceAccess ra )
    {
        this->access_list.erase(ra);
    }

    void build_unique_resource_list()
    {
        for( auto ra = access_list.rbegin(); ra != access_list.rend(); ++ra )
        {
            std::shared_ptr<ResourceBase> r = ra->get_resource();
            unique_resources.erase(ResourceEntry{ r, r->users.rend() });
            unique_resources.push(ResourceEntry{ r, r->users.rend() });
        }
    }

    bool has_sync_access( std::shared_ptr< ResourceBase > res )
    {
        for( auto ra = access_list.rbegin(); ra != access_list.rend(); ++ra )
        {
            if(
               ra->get_resource() == res &&
               ra->is_synchronizing()
            )
                return true;
        }
        return false;
    }

    static bool
    is_serial( ResourceUser const & a, ResourceUser const & b )
    {
        TRACE_EVENT("ResourceUser", "is_serial");
        for( auto ra = a.access_list.crbegin(); ra != a.access_list.crend(); ++ra )
            for( auto rb = b.access_list.crbegin(); rb != b.access_list.crend(); ++rb )
            {
                TRACE_EVENT("ResourceUser", "RA::is_serial");
                if ( ResourceAccess::is_serial( *ra, *rb ) )
                    return true;
            }
        return false;
    }

    bool
    is_superset_of( ResourceUser const & a ) const
    {
        TRACE_EVENT("ResourceUser", "is_superset");
        for( auto ra = a.access_list.rbegin(); ra != a.access_list.rend(); ++ra )
        {
            bool found = false;
            for( auto r = access_list.rbegin(); r != access_list.rend(); ++r )
                if ( r->is_superset_of( *ra ) )
                    found = true;

            if ( !found && ra->scope_level() <= scope_level )
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

    uint8_t scope_level;

    ChunkedList<ResourceAccess> access_list;
    ChunkedList<ResourceEntry> unique_resources;
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

        for( auto it = r.access_list.rbegin(); it != r.access_list.rend(); )
        {
            out = fmt::format_to( out, "{}", *it );
            if( ++it != r.access_list.rend() )
                out = fmt::format_to( out, "," );
        }

        out = fmt::format_to( out, "]" );
        return out;
    }
};

