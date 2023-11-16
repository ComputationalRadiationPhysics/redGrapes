/* Copyright 2019-2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/resource.hpp
 */

#pragma once

#include <boost/type_index.hpp>
#include <memory> // std::unique_ptr<>
#include <vector>
#include <mutex>
#include <iostream>
#include <functional>

#include <redGrapes/task/property/trait.hpp>
#include <redGrapes/memory/allocator.hpp>
#include <redGrapes/util/chunked_list.hpp>
#include <redGrapes/sync/spinlock.hpp>
//#include <redGrapes/dispatch/thread/worker_pool.hpp>
#include <redGrapes_config.hpp>
//#include <redGrapes/redGrapes.hpp>

#include <fmt/format.h>

#ifndef REDGRAPES_RUL_CHUNKSIZE
#define REDGRAPES_RUL_CHUNKSIZE 128
#endif

namespace redGrapes
{

template <typename AccessPolicy>
class Resource;

struct Task;

class ResourceBase
{
protected:
    static unsigned int generateID();

public:
    unsigned int id;
    unsigned int scope_level;

    SpinLock users_mutex;
    ChunkedList< Task* > users;

    /**
     * Create a new resource with an unused ID.
     */
    ResourceBase();

    unsigned get_arena_id() const;
};

template <typename AccessPolicy>
class Resource;

class ResourceAccess
{
    template <typename AccessPolicy>
    friend class Resource;

  private:
    struct AccessBase
    {
        AccessBase( boost::typeindex::type_index access_type, std::shared_ptr<ResourceBase > resource )
            : access_type( access_type )
            , resource( resource )
        {
        }

        AccessBase( AccessBase && other )
            : access_type( other.access_type )
            , resource( std::move(other.resource) )
        {}

        virtual ~AccessBase() {};
        virtual bool operator==( AccessBase const & r ) const = 0;

        bool
        is_same_resource( ResourceAccess::AccessBase const & a ) const
        {
            return this->resource == a.resource;
        }

        virtual bool is_synchronizing() const = 0;
        virtual bool is_serial( AccessBase const & r ) const = 0;
        virtual bool is_superset_of( AccessBase const & r ) const = 0;
        virtual std::string mode_format() const = 0;

        boost::typeindex::type_index access_type;
        std::shared_ptr< ResourceBase > resource;
    }; // AccessBase

    // todo use allocator!!
    std::shared_ptr< AccessBase > obj;

  public:
    ResourceAccess( std::shared_ptr< AccessBase > obj ) : obj( obj ) {}
    ResourceAccess( ResourceAccess const & other ) : obj( other.obj ) {}
    ResourceAccess( ResourceAccess && other ) : obj( std::move(other.obj) ) {
        other.obj.reset();
    }

    ResourceAccess& operator= (ResourceAccess const & other )
    {
        this->obj = other.obj;
        return *this;
    }
    
    static bool
    is_serial( ResourceAccess const & a, ResourceAccess const & b )
    {
        if ( a.obj->access_type == b.obj->access_type )
            return a.obj->is_serial( *b.obj );
        else
            return false;
    }

    bool
    is_superset_of( ResourceAccess const & a ) const
    {
        //if ( this->obj->resource.scope_level < a.obj->resource.scope_level )
        //    return true;
        if ( this->obj->access_type == a.obj->access_type )
            return this->obj->is_superset_of( *a.obj );
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

    std::shared_ptr< ResourceBase > get_resource()
    {
        return obj->resource;
    }
    
    /**
     * Check if the associated resource is the same
     *
     * @param a another ResourceAccess
     * @return true if `a` is associated with the same resource as `this`
     */
    bool
    is_same_resource( ResourceAccess const & a ) const
    {
        if ( this->obj->access_type == a.obj->access_type )
            return this->obj->is_same_resource( *a.obj );
        return false;
    }

    bool
    operator== ( ResourceAccess const & a ) const
    {
        if ( this->obj->access_type == a.obj->access_type )
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
template < typename T >
struct BuildProperties<
    T,
    typename std::enable_if<
        std::is_convertible<T, ResourceAccess>::value
    >::type
>
{
    template < typename Builder >
    inline static void build( Builder & builder, T const & obj )
    {
        builder.add_resource( obj );
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
template <typename AccessPolicy = DefaultAccessPolicy>
class Resource
{
protected:
    struct Access : public ResourceAccess::AccessBase
    {
        Access( std::shared_ptr< ResourceBase > resource, AccessPolicy policy )
            : ResourceAccess::AccessBase(
                  boost::typeindex::type_id<AccessPolicy>(),
                  resource
              )
            , policy( policy )
        {}

        Access( Access && other )
            : AccessBase(std::move((AccessBase&&)other))
            , policy( std::move(other.policy))
        {
        }

        ~Access() {}

        bool is_synchronizing() const
        {
            return policy.is_synchronizing();
        }
        
        bool
        is_serial( ResourceAccess::AccessBase const & a_ ) const
        {
            Access const & a = *static_cast<Access const *>(
                &a_ ); // no dynamic cast needed, type checked in ResourceAccess
            return this->is_same_resource( a ) &&
                   AccessPolicy::is_serial( this->policy, a.policy );
        }

        bool
        is_superset_of( ResourceAccess::AccessBase const & a_ ) const
        {
            Access const & a = *static_cast<Access const *>(
                &a_ ); // no dynamic cast needed, type checked in ResourceAccess
            return this->is_same_resource( a ) &&
                   this->policy.is_superset_of( a.policy );
        }

        bool
        operator==( ResourceAccess::AccessBase const & a_ ) const
        {
            Access const & a = *static_cast<Access const *>(
                &a_ ); // no dynamic cast needed, type checked in ResourceAccess

            return ( this->is_same_resource(a_) && this->policy == a.policy );
        }

        std::string mode_format() const
        {
            return fmt::format("{}", policy);
        }

        AccessPolicy policy;
    }; // struct ThisResourceAccess

    friend class ResourceBase;

    std::shared_ptr< ResourceBase > base;

    Resource( std::shared_ptr<ResourceBase > base )
        : base( base )
    {
    }

  public:
    Resource()
    {
        static unsigned i = 0;

        /* NOTE: Because of #include loops we cannot access Context and thus not worker_pool->size().
         *       for this reason the modulo is done in constructor of Allocator()
         */
        dispatch::thread::WorkerId worker_id = i++; // % SingletonContext::get().worker_pool->size();
        base = redGrapes::memory::alloc_shared_bind< ResourceBase >( worker_id );
    }

    /**
     * Create an ResourceAccess, which represents an concrete
     * access configuration associated with this resource.
     *
     * @param pol AccessPolicy object, containing all access information
     * @return ResourceAccess on this resource
     */
    ResourceAccess
    make_access( AccessPolicy pol ) const
    {
        auto a = redGrapes::memory::alloc_shared_bind< Access >( base->get_arena_id(), base, pol );
        return ResourceAccess( a );
    }
}; // class Resource


template <
    typename T,
    typename AccessPolicy
>
struct SharedResourceObject : Resource< AccessPolicy >
{
    //protected:
    std::shared_ptr< T > obj;

    SharedResourceObject( std::shared_ptr<T> obj )
        : obj(obj) {}

    SharedResourceObject( SharedResourceObject const & other )
        : Resource< AccessPolicy >( other )
        , obj( other.obj )
    {}
}; // struct SharedResourceObject

} // namespace redGrapes

template <>
struct fmt::formatter<
    redGrapes::ResourceAccess
>
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::ResourceAccess const & acc,
        FormatContext & ctx
    )
    {
        return fmt::format_to(
                   ctx.out(),
                   "{{ \"resourceID\" : {}, \"scopeLevel\" : {}, \"mode\" : {} }}",
                   acc.resource_id(),
                   acc.scope_level(),
                   acc.mode_format());
    }
};

