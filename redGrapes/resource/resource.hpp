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

#include "redGrapes/TaskCtx.hpp"
#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/memory/allocator.hpp"
#include "redGrapes/sync/spinlock.hpp"
#include "redGrapes/task/property/trait.hpp"
#include "redGrapes/util/chunked_list.hpp"

#include <boost/type_index.hpp>
#include <fmt/format.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>


#ifndef REDGRAPES_RUL_CHUNKSIZE
#    define REDGRAPES_RUL_CHUNKSIZE 128
#endif

namespace redGrapes
{

    template<typename TTask, typename AccessPolicy>
    class Resource;

    template<typename TTask>
    class ResourceBase
    {
    protected:
        static unsigned int generateID()
        {
            static std::atomic<unsigned int> id_counter;
            return id_counter.fetch_add(1);
        }

    public:
        unsigned int id;
        unsigned int scope_level;

        SpinLock users_mutex;
        ChunkedList<TTask*, REDGRAPES_RUL_CHUNKSIZE> users;

        /**
         * Create a new resource with an unused ID.
         */
        ResourceBase()
            : id(generateID())
            , scope_level(TaskCtx<TTask>::scope_depth())
            , users(memory::Allocator(get_arena_id()))
        {
        }

        unsigned get_arena_id() const
        {
            return id % TaskFreeCtx::n_workers;
        }
    };

    template<typename TTask>
    class ResourceAccess
    {
        // https://stackoverflow.com/questions/16567212/why-does-the-standard-prohibit-friend-declarations-of-partial-specializations
        template<typename TFrenTask, typename AccessPolicy>
        friend class Resource;

    private:
        struct AccessBase
        {
            AccessBase(boost::typeindex::type_index access_type, std::shared_ptr<ResourceBase<TTask>> resource)
                : access_type(access_type)
                , resource(resource)
            {
            }

            AccessBase(AccessBase&& other) : access_type(other.access_type), resource(std::move(other.resource))
            {
            }

            virtual ~AccessBase(){};
            virtual bool operator==(AccessBase const& r) const = 0;

            bool is_same_resource(ResourceAccess<TTask>::AccessBase const& a) const
            {
                return this->resource == a.resource;
            }

            virtual bool is_synchronizing() const = 0;
            virtual bool is_serial(AccessBase const& r) const = 0;
            virtual bool is_superset_of(AccessBase const& r) const = 0;
            virtual std::string mode_format() const = 0;

            boost::typeindex::type_index access_type;
            std::shared_ptr<ResourceBase<TTask>> resource;
        }; // AccessBase

        // todo use allocator!!
        std::shared_ptr<AccessBase> obj;

    public:
        ResourceAccess(std::shared_ptr<AccessBase> obj) : obj(obj)
        {
        }

        ResourceAccess(ResourceAccess<TTask> const& other) : obj(other.obj)
        {
        }

        ResourceAccess(ResourceAccess<TTask>&& other) : obj(std::move(other.obj))
        {
            other.obj.reset();
        }

        ResourceAccess& operator=(ResourceAccess<TTask> const& other)
        {
            this->obj = other.obj;
            return *this;
        }

        static bool is_serial(ResourceAccess<TTask> const& a, ResourceAccess<TTask> const& b)
        {
            if(a.obj->access_type == b.obj->access_type)
                return a.obj->is_serial(*b.obj);
            else
                return false;
        }

        bool is_superset_of(ResourceAccess<TTask> const& a) const
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

        unsigned int resource_id() const
        {
            return this->obj->resource->id;
        }

        std::string mode_format() const
        {
            return this->obj->mode_format();
        }

        std::shared_ptr<ResourceBase<TTask>> get_resource()
        {
            return obj->resource;
        }

        /**
         * Check if the associated resource is the same
         *
         * @param a another ResourceAccess
         * @return true if `a` is associated with the same resource as `this`
         */
        bool is_same_resource(ResourceAccess<TTask> const& a) const
        {
            if(this->obj->access_type == a.obj->access_type)
                return this->obj->is_same_resource(*a.obj);
            return false;
        }

        bool operator==(ResourceAccess<TTask> const& a) const
        {
            if(this->obj->access_type == a.obj->access_type)
                return *(this->obj) == *(a.obj);
            return false;
        }
    }; // class ResourceAccess

    namespace trait
    {

        /**
         * implements BuildProperties for any type which
         * can be casted to a ResourceAccess
         */
        template<typename T, typename TTask>
        struct BuildProperties<
            T,
            TTask,
            typename std::enable_if<std::is_convertible<T, ResourceAccess<TTask>>::value>::type>
        {
            template<typename Builder>
            static inline void build(Builder& builder, T const& obj)
            {
                builder.add_resource(obj);
            }
        };
    } // namespace trait

    struct DefaultAccessPolicy
    {
        static bool is_serial(DefaultAccessPolicy, DefaultAccessPolicy)
        {
            return true;
        }
    };

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
    template<typename TTask, typename AccessPolicy = DefaultAccessPolicy>
    class Resource
    {
    protected:
        struct Access : public ResourceAccess<TTask>::AccessBase
        {
            Access(std::shared_ptr<ResourceBase<TTask>> resource, AccessPolicy policy)
                : ResourceAccess<TTask>::AccessBase(boost::typeindex::type_id<AccessPolicy>(), resource)
                , policy(policy)
            {
            }

            Access(Access&& other)
                : ResourceAccess<TTask>::AccessBase(
                    std::move(std::forward<ResourceAccess<TTask>::AccessBase>(other))) // TODO check this
                , policy(std::move(other.policy))
            {
            }

            ~Access()
            {
            }

            bool is_synchronizing() const
            {
                return policy.is_synchronizing();
            }

            bool is_serial(typename ResourceAccess<TTask>::AccessBase const& a_) const
            {
                Access const& a
                    = *static_cast<Access const*>(&a_); // no dynamic cast needed, type checked in ResourceAccess
                return this->is_same_resource(a) && AccessPolicy::is_serial(this->policy, a.policy);
            }

            bool is_superset_of(typename ResourceAccess<TTask>::AccessBase const& a_) const
            {
                Access const& a
                    = *static_cast<Access const*>(&a_); // no dynamic cast needed, type checked in ResourceAccess
                return this->is_same_resource(a) && this->policy.is_superset_of(a.policy);
            }

            bool operator==(typename ResourceAccess<TTask>::AccessBase const& a_) const
            {
                Access const& a
                    = *static_cast<Access const*>(&a_); // no dynamic cast needed, type checked in ResourceAccess

                return (this->is_same_resource(a_) && this->policy == a.policy);
            }

            std::string mode_format() const
            {
                return fmt::format("{}", policy);
            }

            AccessPolicy policy;
        }; // struct ThisResourceAccess

        friend class ResourceBase<TTask>;

        std::shared_ptr<ResourceBase<TTask>> base;

        Resource(std::shared_ptr<ResourceBase<TTask>> base) : base(base)
        {
        }

    public:
        Resource()
        {
            static unsigned i = 0;

            WorkerId worker_id = i++ % TaskFreeCtx::n_workers;
            base = redGrapes::memory::alloc_shared_bind<ResourceBase<TTask>>(worker_id);
        }

        /**
         * Create an ResourceAccess, which represents an concrete
         * access configuration associated with this resource.
         *
         * @param pol AccessPolicy object, containing all access information
         * @return ResourceAccess on this resource
         */
        ResourceAccess<TTask> make_access(AccessPolicy pol) const
        {
            auto a = redGrapes::memory::alloc_shared_bind<Access>(base->get_arena_id(), base, pol);
            return ResourceAccess<TTask>(a);
        }
    }; // class Resource

    template<typename T, typename TTask, typename AccessPolicy>
    struct SharedResourceObject : Resource<TTask, AccessPolicy>
    {
        // protected:
        std::shared_ptr<T> obj;

        SharedResourceObject(std::shared_ptr<T> obj) : obj(obj)
        {
        }

        SharedResourceObject(SharedResourceObject const& other) : Resource<TTask, AccessPolicy>(other), obj(other.obj)
        {
        }
    }; // struct SharedResourceObject

} // namespace redGrapes

template<typename TTask>
struct fmt::formatter<redGrapes::ResourceAccess<TTask>>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::ResourceAccess<TTask> const& acc, FormatContext& ctx)
    {
        return fmt::format_to(
            ctx.out(),
            "{{ \"resourceID\" : {}, \"scopeLevel\" : {}, \"mode\" : {} }}",
            acc.resource_id(),
            acc.scope_level(),
            acc.mode_format());
    }
};
