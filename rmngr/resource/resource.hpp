
/**
 * @file rmngr/resource.hpp
 */

#pragma once

#include <boost/type_index.hpp>
#include <memory> // std::unique_ptr<>
#include <vector>
#include <atomic>
#include <iostream>

namespace rmngr
{

template <typename AccessPolicy>
class Resource;

class ResourceAccess
{
    template <typename AccessPolicy>
    friend class Resource;

  private:
    struct AccessBase
    {
        AccessBase( boost::typeindex::type_index access_type_ )
            : access_type( access_type_ )
        {
        }

        virtual ~AccessBase() {};
        virtual bool operator==( AccessBase const & r ) const = 0;
        virtual bool is_same_resource( AccessBase const & r ) const = 0;
        virtual bool is_serial( AccessBase const & r ) const = 0;
        virtual bool is_superset_of( AccessBase const & r ) const = 0;
        virtual AccessBase * clone( void ) const = 0;
        virtual std::ostream& write(std::ostream&) = 0;

        boost::typeindex::type_index access_type;
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
        if ( this->obj->access_type == a.obj->access_type )
            return this->obj->is_superset_of( *a.obj );
        else
            return false;
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
        Access( Resource<AccessPolicy> resource_, AccessPolicy policy_ )
            : ResourceAccess::AccessBase(
                  boost::typeindex::type_id<AccessPolicy>() ),
              resource( resource_ ),
              policy( policy_ )
        {
        }

        ~Access() {}

        bool
        is_same_resource( ResourceAccess::AccessBase const & a_ ) const
        {
            Access const & a = *static_cast<Access const *>(
                &a_ ); // no dynamic cast needed, type checked in ResourceAccess
            return ( this->resource.id == a.resource.id );
        }

        bool
        is_serial( ResourceAccess::AccessBase const & a_ ) const
        {
            Access const & a = *static_cast<Access const *>(
                &a_ ); // no dynamic cast needed, type checked in ResourceAccess
            return ( this->is_same_resource( a ) ) &&
                   ( AccessPolicy::is_serial( this->policy, a.policy ) );
        }

        bool
        is_superset_of( ResourceAccess::AccessBase const & a_ ) const
        {
            Access const & a = *static_cast<Access const *>(
                &a_ ); // no dynamic cast needed, type checked in ResourceAccess
            return ( this->is_same_resource( a ) ) &&
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
            out << "Resource(" << resource.id << ")::";
	    out << policy;
            return out;
        }

        Resource<AccessPolicy> resource;
        AccessPolicy policy;
    }; // struct ThisResourceAccess

    unsigned int const id;
    static std::atomic_int id_counter;

  public:
    /**
     * Create a new resource with an unused ID.
     */
    Resource()
      : id( ++id_counter )
    {}

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

template <typename AccessPolicy>
std::atomic_int Resource<AccessPolicy>::id_counter;

} // namespace rmngr
