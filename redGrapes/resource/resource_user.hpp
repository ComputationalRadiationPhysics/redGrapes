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

#include <redGrapes/memory/allocator.hpp>
#include <redGrapes/util/chunked_list.hpp>
#include <redGrapes/util/trace.hpp>

//#include <redGrapes/resource/resource.hpp>

namespace redGrapes
{

unsigned scope_depth();

struct Task;
struct ResourceBase;
struct ResourceAccess;

struct ResourceUsageEntry
{
    std::shared_ptr< ResourceBase > resource;
    typename ChunkedList< Task* >::MutBackwardIterator task_entry;

    bool operator==( ResourceUsageEntry const & other ) const;
};

class ResourceUser
{
  public:    
    ResourceUser();
    ResourceUser( ResourceUser const& other );
    ResourceUser( std::initializer_list< ResourceAccess > list );
 
    void add_resource_access( ResourceAccess ra );
    void rm_resource_access( ResourceAccess ra );
    void build_unique_resource_list();
    bool has_sync_access( std::shared_ptr< ResourceBase > res );
    bool is_superset_of( ResourceUser const & a ) const;
    static bool is_superset( ResourceUser const & a, ResourceUser const & b );   
    static bool is_serial( ResourceUser const & a, ResourceUser const & b );

    uint8_t scope_level;

    ChunkedList<ResourceAccess> access_list;
    ChunkedList<ResourceUsageEntry> unique_resources;
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

