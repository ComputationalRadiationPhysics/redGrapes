/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/resource.hpp
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/memory/allocator.hpp"
#include "redGrapes/sync/spinlock.hpp"
#include "redGrapes/task/property/trait.hpp"
#include "redGrapes/util/chunked_list.hpp"
#include "redGrapes/util/traits.hpp"

#include <boost/type_index.hpp>
#include <fmt/format.h>

#include <cassert>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>


#ifndef REDGRAPES_RUL_CHUNKSIZE
#    define REDGRAPES_RUL_CHUNKSIZE 128
#endif

namespace redGrapes
{

    struct ResourceUser;
    unsigned scope_depth_impl();

    namespace mapping
    {
        template<typename MappingFunc>
        struct MapResourceIdToWorker
        {
            MappingFunc mappingFunc;

            WorkerId operator()(ResourceId resourceId) const
            {
                return mappingFunc(resourceId);
            }
        };

        struct ModuloMapping
        {
            WorkerId operator()(ResourceId resourceId) const
            {
                return resourceId % TaskFreeCtx::n_workers;
            }
        };

        static const MapResourceIdToWorker<ModuloMapping> map_resource_to_worker{};

    } // namespace mapping

    template<typename AccessPolicy>
    class Resource;

    class ResourceBase
    {
    public:
        ChunkedList<ResourceUser*, REDGRAPES_RUL_CHUNKSIZE> users;
        SpinLock users_mutex;
        ResourceId id;
        uint8_t scope_level;

        /**
         * Create a new resource with an unused ID.
         */
        ResourceBase(ResourceId id)
            : users(memory::Allocator(mapping::map_resource_to_worker(id)))
            , id(id)
            , scope_level(scope_depth_impl())
        {
        }
    };

    struct AccessBase
    {
        AccessBase(boost::typeindex::type_index access_type, std::shared_ptr<ResourceBase> resource)
            : access_type(access_type)
            , resource(std::move(resource))
        {
        }

        AccessBase(AccessBase&& other) : access_type(other.access_type), resource(std::move(other.resource))
        {
        }

        virtual ~AccessBase(){};
        virtual bool operator==(AccessBase const& r) const = 0;

        bool is_same_resource(AccessBase const& a) const
        {
            return this->resource == a.resource;
        }

        virtual bool is_synchronizing() const = 0;
        virtual bool is_serial(AccessBase const& r) const = 0;
        virtual bool is_superset_of(AccessBase const& r) const = 0;
        virtual std::string mode_format() const = 0;

        boost::typeindex::type_index access_type;
        std::shared_ptr<ResourceBase> resource;
    }; // AccessBase

    template<typename AccessPolicy>
    struct Access : public AccessBase
    {
        Access(std::shared_ptr<ResourceBase> const& resource, AccessPolicy policy)
            : AccessBase(boost::typeindex::type_id<AccessPolicy>(), resource)
            , policy(policy)
        {
        }

        Access(Access&& other) noexcept
            : AccessBase(std::move(std::forward<AccessBase>(other))) // TODO check this
            , policy(std::move(other.policy))
        {
        }

        ~Access() override = default;

        bool is_synchronizing() const override
        {
            return policy.is_synchronizing();
        }

        bool is_serial(AccessBase const& a_) const override
        {
            Access const& a
                = *static_cast<Access const*>(&a_); // no dynamic cast needed, type checked in ResourceAccess
            return this->is_same_resource(a) && AccessPolicy::is_serial(this->policy, a.policy);
        }

        bool is_superset_of(AccessBase const& a_) const override
        {
            Access const& a
                = *static_cast<Access const*>(&a_); // no dynamic cast needed, type checked in ResourceAccess
            return this->is_same_resource(a) && this->policy.is_superset_of(a.policy);
        }

        bool operator==(AccessBase const& a_) const override
        {
            Access const& a
                = *static_cast<Access const*>(&a_); // no dynamic cast needed, type checked in ResourceAccess

            return (this->is_same_resource(a_) && this->policy == a.policy);
        }

        std::string mode_format() const override
        {
            return fmt::format("{}", policy);
        }

        AccessPolicy policy;
    }; // struct Access

    class ResourceAccess
    {
    private:
        // todo use allocator!!
        std::shared_ptr<AccessBase> obj;

    public:
        ResourceAccess(std::shared_ptr<AccessBase> obj) : obj(std::move(obj))
        {
        }

        ResourceAccess(ResourceAccess const& other) : obj(other.obj)
        {
        }

        ResourceAccess(ResourceAccess&& other) : obj(std::move(other.obj))
        {
            other.obj.reset();
        }

        ResourceAccess& operator=(ResourceAccess const& other)
        {
            this->obj = other.obj;
            return *this;
        }

        template<typename AccessPolicy>
        friend ResourceAccess newAccess(ResourceAccess const& x, AccessPolicy pol)
        {
            assert(x.obj->access_type == boost::typeindex::type_id<AccessPolicy>());
            auto y = redGrapes::memory::alloc_shared_bind<Access<AccessPolicy>>(
                mapping::map_resource_to_worker(x.obj->resource->id),
                x.obj->resource,
                pol);
            return ResourceAccess(y);
        }

        friend bool is_serial(ResourceAccess const& a, ResourceAccess const& b)
        {
            if(a.obj->access_type == b.obj->access_type)
                return a.obj->is_serial(*b.obj);
            else
                return false;
        }

        bool is_superset_of(ResourceAccess const& a) const
        {
            // if ( this->obj->resource.scope_level < a.obj->resource.scope_level )
            //     return true;
            if(this->obj->access_type == a.obj->access_type)
                return this->obj->is_superset_of(*a.obj);
            else
                return false;
        }

        bool is_synchronizing() const
        {
            return this->obj->is_synchronizing();
        }

        unsigned int scope_level() const
        {
            return this->obj->resource->scope_level;
        }

        ResourceId resource_id() const
        {
            return this->obj->resource->id;
        }

        std::string mode_format() const
        {
            return this->obj->mode_format();
        }

        // Doesn't share ownership
        ResourceBase* get_resource_ptr() const
        {
            return (obj->resource).get();
        }

        /**
         * Check if the associated resource is the same
         *
         * @param a another ResourceAccess
         * @return true if `a` is associated with the same resource as `this`
         */
        bool is_same_resource(ResourceAccess const& a) const
        {
            if(this->obj->access_type == a.obj->access_type)
                return this->obj->is_same_resource(*a.obj);
            return false;
        }

        bool operator==(ResourceAccess const& a) const
        {
            if(this->obj->access_type == a.obj->access_type)
                return *(this->obj) == *(a.obj);
            return false;
        }
    }; // class ResourceAccess

    template<typename AccessPolicy>
    ResourceAccess newAccess(ResourceAccess const& x, AccessPolicy pol);

    bool is_serial(ResourceAccess const& a, ResourceAccess const& b);

    template<typename T>
    struct ResourceAccessPair : public std::pair<T, ResourceAccess>
    {
        using std::pair<T, ResourceAccess>::pair;

        operator ResourceAccess() const
        {
            return this->second;
        }

    }; // namespace std::pair

    namespace trait
    {

        /**
         * implements BuildProperties for any type which
         * can be casted to a ResourceAccess
         */
        template<typename T, typename TTask>
        struct BuildProperties<T, TTask, typename std::enable_if<std::is_convertible<T, ResourceAccess>::value>::type>
        {
            template<typename Builder>
            static inline void build(Builder& builder, T const& obj)
            {
                builder.add_resource(obj);
            }
        };
    } // namespace trait

    namespace access
    {
        struct DefaultAccessPolicy
        {
            static bool is_serial(DefaultAccessPolicy, DefaultAccessPolicy)
            {
                return true;
            }
        };
    } // namespace access

    /**
     * @defgroup AccessPolicy
     *
     * @{
     *
     * @par Description
     * An implementation of the concept AccessPolicy creates a new resource-type (`Resource<AccessPolicy>`)
     * and should define the possible access modes / configurations for this resource-type (e.g. read/write)
     *
     * @par Required public member functions
     * - `static bool is_serial(AccessPolicy, AccessPolicy)`
     * check if the two accesses have to be **in order**. (e.g. two reads return false, an occuring write always true)
     *
     * - `static bool is_superset(AccessPolicy a, AccessPolicy b)`
     * check if access `a` is a superset of access `b` (e.g. accessing [0,3] is a superset of accessing [1,2])
     *
     * @}
     */

    /**
     * @class Resource
     * @tparam AccessPolicy Defines the access-modes (e.g. read/write) that are possible
     *                      with this resource. Required to implement the concept @ref AccessPolicy
     *
     * Represents a concrete resource.
     * Copied objects represent the same resource.
     */
    template<typename AccessPolicy = access::DefaultAccessPolicy>
    class Resource
    {
    protected:
        std::shared_ptr<ResourceBase> base;

    public:
        Resource(ResourceId id)
            : base{redGrapes::memory::alloc_shared_bind<ResourceBase>(mapping::map_resource_to_worker(id), id)}

        {
        }

        /**
         * Create an ResourceAccess, which represents an concrete
         * access configuration associated with this resource.
         *
         * @param pol AccessPolicy object, containing all access information
         * @return ResourceAccess on this resource
         */
        ResourceAccess make_access(AccessPolicy pol) const
        {
            auto a = redGrapes::memory::alloc_shared_bind<Access<AccessPolicy>>(
                mapping::map_resource_to_worker(base->id),
                base,
                pol);
            return ResourceAccess(a);
        }

        ResourceId resource_id() const
        {
            return base->id;
        }


    }; // class Resource

    template<typename T, typename AccessPolicy>
    struct SharedResourceObject
    {
        SharedResourceObject(std::shared_ptr<T> obj) : res{TaskFreeCtx::create_resource_uid()}, obj(std::move(obj))
        {
        }

        template<typename... Args>
        requires(!(traits::is_specialization_of_v<std::decay_t<traits::first_type_t<Args...>>, SharedResourceObject>
                   || std::is_same_v<std::decay_t<traits::first_type_t<Args...>>, std::shared_ptr<T>>) )
        SharedResourceObject(Args&&... args)
            : res{TaskFreeCtx::create_resource_uid()}
            , obj{memory::alloc_shared_bind<T>(
                  mapping::map_resource_to_worker(res.resource_id()),
                  std::forward<Args>(args)...)}
        {
        }

        template<typename U>
        SharedResourceObject(SharedResourceObject<U, AccessPolicy> const& res, std::shared_ptr<T> obj)
            : res{res}
            , obj{std::move(obj)}
        {
        }

        template<typename U, typename... Args>
        requires(!(std::is_same_v<std::decay_t<traits::first_type_t<Args...>>, std::shared_ptr<T>>) )
        SharedResourceObject(SharedResourceObject<U, AccessPolicy> const& res, Args&&... args)
            : res{res}
            , obj{memory::alloc_shared_bind<T>(
                  mapping::map_resource_to_worker(res.resource_id()),
                  std::forward<Args>(args)...)}
        {
        }

        // DANGER! Gives access to ibject being held as a resource without any synchronization and scheduling
        // Only to be used when you know what you are doing
        auto& getObject()
        {
            return obj;
        }

    protected:
        Resource<AccessPolicy> res;
        std::shared_ptr<T> obj;

    }; // struct SharedResourceObject

} // namespace redGrapes

template<>
struct fmt::formatter<redGrapes::ResourceAccess>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::ResourceAccess const& acc, FormatContext& ctx) const
    {
        return fmt::format_to(
            ctx.out(),
            "{{ \"resourceID\" : {}, \"scopeLevel\" : {}, \"mode\" : {} }}",
            acc.resource_id(),
            acc.scope_level(),
            acc.mode_format());
    }
};
