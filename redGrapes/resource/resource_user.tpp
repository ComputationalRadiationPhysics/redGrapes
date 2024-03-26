/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once
#include "redGrapes/TaskCtx.hpp"
#include "redGrapes/resource/resource.hpp"
#include "redGrapes/resource/resource_user.hpp"
#include "redGrapes/util/trace.hpp"

namespace redGrapes
{
    template<typename TTask>
    bool ResourceUsageEntry<TTask>::operator==(ResourceUsageEntry<TTask> const& other) const
    {
        return resource == other.resource;
    }

    template<typename TTask>
    ResourceUser<TTask>::ResourceUser()
        : scope_level(TaskCtx<TTask>::scope_depth())
        , access_list(memory::Allocator())
        , unique_resources(memory::Allocator())
    {
    }

    template<typename TTask>
    ResourceUser<TTask>::ResourceUser(ResourceUser<TTask> const& other)
        : scope_level(other.scope_level)
        , access_list(memory::Allocator(), other.access_list)
        , unique_resources(memory::Allocator(), other.unique_resources)
    {
    }

    template<typename TTask>
    ResourceUser<TTask>::ResourceUser(std::initializer_list<ResourceAccess<TTask>> list)
        : scope_level(TaskCtx<TTask>::scope_depth())
        , access_list(memory::Allocator())
        , unique_resources(memory::Allocator())
    {
        for(auto& ra : list)
            add_resource_access(ra);
    }

    template<typename TTask>
    void ResourceUser<TTask>::add_resource_access(ResourceAccess<TTask> ra)
    {
        this->access_list.push(ra);
        std::shared_ptr<ResourceBase<TTask>> r = ra.get_resource();
        unique_resources.push(ResourceUsageEntry<TTask>{r, r->users.rend()});
    }

    template<typename TTask>
    void ResourceUser<TTask>::rm_resource_access(ResourceAccess<TTask> ra)
    {
        this->access_list.erase(ra);
    }

    template<typename TTask>
    void ResourceUser<TTask>::build_unique_resource_list()
    {
        for(auto ra = access_list.rbegin(); ra != access_list.rend(); ++ra)
        {
            std::shared_ptr<ResourceBase<TTask>> r = ra->get_resource();
            unique_resources.erase(ResourceUsageEntry<TTask>{r, r->users.rend()});
            unique_resources.push(ResourceUsageEntry<TTask>{r, r->users.rend()});
        }
    }

    template<typename TTask>
    bool ResourceUser<TTask>::has_sync_access(std::shared_ptr<ResourceBase<TTask>> const& res)
    {
        for(auto ra = access_list.rbegin(); ra != access_list.rend(); ++ra)
        {
            if(ra->get_resource() == res && ra->is_synchronizing())
                return true;
        }
        return false;
    }

    template<typename TTask>
    bool ResourceUser<TTask>::is_serial(ResourceUser<TTask> const& a, ResourceUser<TTask> const& b)
    {
        TRACE_EVENT("ResourceUser", "is_serial");
        for(auto ra = a.access_list.crbegin(); ra != a.access_list.crend(); ++ra)
            for(auto rb = b.access_list.crbegin(); rb != b.access_list.crend(); ++rb)
            {
                TRACE_EVENT("ResourceUser", "RA::is_serial");
                if(ResourceAccess<TTask>::is_serial(*ra, *rb))
                    return true;
            }
        return false;
    }

    template<typename TTask>
    bool ResourceUser<TTask>::is_superset_of(ResourceUser<TTask> const& a) const
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

    template<typename TTask>
    bool ResourceUser<TTask>::is_superset(ResourceUser<TTask> const& a, ResourceUser<TTask> const& b)
    {
        return a.is_superset_of(b);
    }

} // namespace redGrapes
