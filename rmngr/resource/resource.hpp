
/**
 * @file rmngr/resource.hpp
 */

#pragma once

#include <boost/type_index.hpp>
#include <memory> // std::unique_ptr<>
#include <vector>

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

        virtual ~AccessBase(){};
        virtual bool is_same_resource( AccessBase const & r ) const = 0;
        virtual bool is_serial( AccessBase const & r ) const = 0;
        virtual bool is_superset_of( AccessBase const & r ) const = 0;
        virtual AccessBase * clone( void ) const = 0;
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

    bool
    is_same_resource( ResourceAccess const & a ) const
    {
        if ( this->obj->access_type == a.obj->access_type )
            return this->obj->is_same_resource( *a.obj );
        return false;
    }
}; // class ResourceAccess

struct DefaultAccessPolicy
{
    static bool is_serial(DefaultAccessPolicy, DefaultAccessPolicy)
    {
        return true;
    }
};

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

        AccessBase *
        clone( void ) const
        {
            return new Access( this->resource, this->policy );
        }

        Resource<AccessPolicy> resource;
        AccessPolicy policy;
    }; // struct ThisResourceAccess

    unsigned int const id;
    static unsigned int id_counter;

  public:
    Resource() : id( id_counter++ ) {}

    ResourceAccess
    make_access( AccessPolicy pol ) const
    {
        return ResourceAccess( new Access( *this, pol ) );
    }
}; // class Resource

template <typename AccessPolicy>
unsigned int Resource<AccessPolicy>::id_counter = 0;

} // namespace rmngr
