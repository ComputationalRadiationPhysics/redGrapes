/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/resource/resource_user.hpp
 */

#pragma once

#include "redGrapes/resource/resource.hpp"
#include "redGrapes/util/chunked_list.hpp"

#include <fmt/format.h>

#include <initializer_list>
#include <memory>

namespace redGrapes
{
#ifndef REDGRAPES_RUL_CHUNKSIZE
#    define REDGRAPES_RUL_CHUNKSIZE 128
#endif

    struct TaskSpace;

    // unique_resources - list of ResourceUsageEntry Doesnt own Resource Base.
    // access_list - list of ResourceAccess is responsible for resource base lifetime
    struct ResourceUsageEntry
    {
        ResourceBase* resource;
        typename ChunkedList<ResourceUser*, REDGRAPES_RUL_CHUNKSIZE>::MutBackwardIterator user_entry;

        friend bool operator==(ResourceUsageEntry const& a, ResourceUsageEntry const& b)
        {
            return a.resource == b.resource;
        }
    };

    struct ResourceUser
    {
        ResourceUser(WorkerId worker_id, unsigned scope_depth)
            : access_list(memory::Allocator(worker_id))
            , unique_resources(memory::Allocator(worker_id))
            , scope_level(scope_depth)
        {
        }

        ResourceUser(ResourceUser const& other) = delete;

        ResourceUser(ResourceUser const& other, WorkerId worker_id)
            : access_list(memory::Allocator(worker_id), other.access_list)
            , unique_resources(memory::Allocator(worker_id), other.unique_resources)
            , scope_level(other.scope_level)
        {
        }

        ResourceUser(std::initializer_list<ResourceAccess> list, WorkerId worker_id, unsigned scope_depth)
            : access_list(memory::Allocator(worker_id))
            , unique_resources(memory::Allocator(worker_id))
            , scope_level(scope_depth)
        {
            for(auto& ra : list)
                add_resource_access(ra);
        }

        void add_resource_access(ResourceAccess const& ra)
        {
            this->access_list.push(ra);
            auto* r = ra.get_resource_ptr();
            unique_resources.push(ResourceUsageEntry{r, r->users.rend()});
        }

        void rm_resource_access(ResourceAccess const& ra)
        {
            this->access_list.erase(ra);
        }

        void build_unique_resource_list()
        {
            for(auto ra = access_list.rbegin(); ra != access_list.rend(); ++ra)
            {
                auto* r = ra->get_resource_ptr();
                unique_resources.erase(ResourceUsageEntry{r, r->users.rend()});
                unique_resources.push(ResourceUsageEntry{r, r->users.rend()});
            }
        }

        bool has_sync_access(ResourceBase* res)
        {
            for(auto ra = access_list.rbegin(); ra != access_list.rend(); ++ra)
            {
                if(ra->get_resource_ptr() == res && ra->is_synchronizing())
                    return true;
            }
            return false;
        }

        bool is_superset_of(ResourceUser const& a) const
        {
            TRACE_EVENT("ResourceUser", "is_superset");
            for(auto ra = a.access_list.rbegin(); ra != a.access_list.rend(); ++ra)
            {
                bool found = false;
                for(auto r = access_list.rbegin(); r != access_list.rend(); ++r)
                    if(r->is_superset_of(*ra))
                        found = true;

                if(!found && ra->scope_level() <= scope_level)
                    // a introduced a new resource
                    return false;
            }
            return true;
        }

        friend bool is_serial(ResourceUser const& a, ResourceUser const& b)
        {
            TRACE_EVENT("ResourceUser", "is_serial");
            for(auto ra = a.access_list.crbegin(); ra != a.access_list.crend(); ++ra)
                for(auto rb = b.access_list.crbegin(); rb != b.access_list.crend(); ++rb)
                {
                    TRACE_EVENT("ResourceUser", "RA::is_serial");
                    if(is_serial(*ra, *rb))
                        return true;
                }
            return false;
        }

        ChunkedList<ResourceAccess, 8> access_list;
        ChunkedList<ResourceUsageEntry, 8> unique_resources;
        uint8_t scope_level;
        //! task space that contains this task, must not be null
        std::shared_ptr<TaskSpace> space;

        //! task space for children, may be null
        std::shared_ptr<TaskSpace> children;

    }; // struct ResourceUser

} // namespace redGrapes

template<>
struct fmt::formatter<redGrapes::ResourceUser>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::ResourceUser const& r, FormatContext& ctx) const
    {
        auto out = ctx.out();
        out = fmt::format_to(out, "[");

        for(auto it = r.access_list.rbegin(); it != r.access_list.rend();)
        {
            out = fmt::format_to(out, "{}", *it);
            if(++it != r.access_list.rend())
                out = fmt::format_to(out, ",");
        }

        out = fmt::format_to(out, "]");
        return out;
    }
};
