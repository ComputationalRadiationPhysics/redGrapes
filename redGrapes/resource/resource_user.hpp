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

#include "redGrapes/util/chunked_list.hpp"

#include <fmt/format.h>

#include <initializer_list>
#include <memory>

namespace redGrapes
{
#ifndef REDGRAPES_RUL_CHUNKSIZE
#    define REDGRAPES_RUL_CHUNKSIZE 128
#endif


    unsigned scope_depth();

    template<typename TTask>
    struct ResourceBase;

    template<typename TTask>
    struct ResourceAccess;

    template<typename TTask>
    struct ResourceUsageEntry
    {
        std::shared_ptr<ResourceBase<TTask>> resource;
        typename ChunkedList<TTask*, REDGRAPES_RUL_CHUNKSIZE>::MutBackwardIterator task_entry;

        bool operator==(ResourceUsageEntry<TTask> const& other) const;
    };

    template<typename TTask>
    class ResourceUser
    {
    public:
        ResourceUser();
        ResourceUser(ResourceUser const& other);
        ResourceUser(std::initializer_list<ResourceAccess<TTask>> list);

        void add_resource_access(ResourceAccess<TTask> ra);
        void rm_resource_access(ResourceAccess<TTask> ra);
        void build_unique_resource_list();
        bool has_sync_access(std::shared_ptr<ResourceBase<TTask>> const& res);
        bool is_superset_of(ResourceUser const& a) const;
        static bool is_superset(ResourceUser const& a, ResourceUser const& b);
        static bool is_serial(ResourceUser const& a, ResourceUser const& b);

        uint8_t scope_level;

        ChunkedList<ResourceAccess<TTask>, 8> access_list;
        ChunkedList<ResourceUsageEntry<TTask>, 8> unique_resources;
    }; // class ResourceUser

} // namespace redGrapes

template<typename TTask>
struct fmt::formatter<redGrapes::ResourceUser<TTask>>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::ResourceUser<TTask> const& r, FormatContext& ctx)
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
