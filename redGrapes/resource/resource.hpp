/* Copyright 2019 Michael Sippel
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

#include <redGrapes/thread_local.hpp>

namespace redGrapes
{

class ResourceBase
{
protected:
    static int getID()
    {
        static std::mutex m;
        static int id_counter;
        std::lock_guard<std::mutex> lock(m);
        return id_counter ++;
    }

public:
    unsigned int id;
    unsigned int scope_level;

    /**
     * Create a new resource with an unused ID.
     */
    ResourceBase()
        : id( getID() )
        , scope_level( thread::scope_level )
    {}
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
        AccessBase( boost::typeindex::type_index access_type_, ResourceBase resource )
            : access_type( access_type_ )
            , resource( resource )
        {
        }

        virtual ~AccessBase() {};
        virtual bool operator==( AccessBase const & r ) const = 0;

        bool
        is_same_resource( ResourceAccess::AccessBase const & a ) const
        {
            return this->resource.id == a.resource.id;
        }

        virtual bool is_serial( AccessBase const & r ) const = 0;
        virtual bool is_superset_of( AccessBase const & r ) const = 0;
        virtual AccessBase * clone( void ) const = 0;
        virtual std::ostream& write(std::ostream&) = 0;

        boost::typeindex::type_index access_type;
        ResourceBase resource;
    }; // AccessBase

    std::unique_ptr<AccessBase> obj;

  public:
    ResourceAccess( AccessBase * obj_ ) : obj( obj_ ) {}
    ResourceAccess( ResourceAccess const & r ) : obj( r.obj->clone() ) {}
    ResourceAccess( ResourceAccess && r ) : obj( std::move( r.obj ) ) {}

    ResourceAccess &
    operator=( ResourceAccess const & r )
    {
        this->obj.reset( r.obj->clone() );
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

    unsigned int scope_level() const
    {
        return this->obj->resource.scope_level;
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

    friend std::ostream& operator<<(std::ostream& out, ResourceAccess const & acc)
    {
        return acc.obj->write(out);
    }
}; // class ResourceAccess

/**
 * implements BuildProperties for a type "name" which
 * can be casted to a ResourceAccess
 */
#define TRAIT_BUILD_RESOURCE_PROPERTIES( name )                  \
    struct BuildProperties< name >                               \
    {                                                            \
        template < typename Builder >                            \
        static void build( Builder & builder, name const & x )   \
        {                                                        \
            builder.add_resource( x );                           \
        }                                                        \
    };                                                           \

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
class Resource : public ResourceBase
{
protected:
    struct Access : public ResourceAccess::AccessBase
    {
        Access( ResourceBase resource_, AccessPolicy policy_ )
            : ResourceAccess::AccessBase(
                  boost::typeindex::type_id<AccessPolicy>(),
                  resource_
              )
            , policy( policy_ )
        {}

        ~Access() {}

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

        AccessBase *
        clone( void ) const
        {
            return new Access( this->resource, this->policy );
        }

        std::ostream& write(std::ostream& out)
        {
            out << "Resource(" << this->resource.id << " [" << this->resource.scope_level << "])::";
	    out << policy;
            return out;
        }

        AccessPolicy policy;
    }; // struct ThisResourceAccess

  public:
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
        return ResourceAccess( new Access( *this, pol ) );
    }
}; // class Resource


template <
    typename T,
    typename AccessPolicy
>
struct SharedResourceObject : Resource< AccessPolicy >
{
protected:
    std::shared_ptr< T > obj;

    SharedResourceObject( std::shared_ptr<T> obj )
        : obj(obj)
    {}

    SharedResourceObject( SharedResourceObject const & other )
        : Resource< AccessPolicy >( other )
        , obj( other.obj )
    {}
}; // struct SharedResourceObject

} // namespace redGrapes
