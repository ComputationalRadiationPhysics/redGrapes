/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/resource/ioresource.hpp
 */

#pragma once

#include <redGrapes/access/io.hpp>
#include <redGrapes/resource/resource.hpp>
#include <redGrapes/property/resource.hpp>

#include <redGrapes/property/trait.hpp>

namespace redGrapes
{

template <typename T>
struct IOResource;

namespace ioresource
{

template <typename T>
struct Guard
{
protected:
    friend class IOResource<T>;

    std::shared_ptr< T > obj;
    redGrapes::Resource< access::IOAccess > resource;

    Guard( std::shared_ptr<T> obj, redGrapes::Resource< access::IOAccess > resource )
        : obj(obj)
        , resource(resource)
    {}

    Guard( Guard const & other )
        : obj(other.obj)
        , resource(other.resource)
    {}
};

template <typename T>
struct ReadGuard : Guard<T>
{
    ReadGuard read() const noexcept { return *this; }
    T const & operator* () const noexcept { return *this->obj; }
    T const * operator-> () const noexcept { return this->obj.get(); }

    operator ResourceAccess () const
    {
        return this->resource.make_access( access::IOAccess::read );
    }

    template <typename Builder>
    void build_properties( Builder & builder ) const
    {
        builder.add_resource( *this );
    }

protected:
    friend class IOResource<T>;
    ReadGuard(Guard<T> const & other) : Guard<T>(other) {}
};

template <typename T>
struct WriteGuard : ReadGuard<T>
{
    WriteGuard write() const noexcept { return *this; }
    T & operator* () const noexcept { return *this->obj; }
    T * operator-> () const noexcept { return this->obj.get(); }

    operator ResourceAccess () const
    {
        return this->resource.make_access( access::IOAccess::write );
    }

    template <typename B>
    void build_properties( ResourceProperty::Builder<B> & builder ) const
    {
        builder.add_resource( *this );
    }

protected:
    friend class IOResource<T>;
    WriteGuard(Guard<T> const & other) : ReadGuard<T>(other) {}
};

} // namespace ioresource

template < typename T >
struct IOResource
{
    using Guard = ioresource::Guard<T>;
    using ReadGuard = ioresource::ReadGuard<T>;
    using WriteGuard = ioresource::WriteGuard<T>;

    template < typename... Args >
    IOResource( Args&&... args )
        : g(
              std::make_shared<T>(std::forward<Args>(args)...),
              Resource<access::IOAccess>()
          )
    {}

    auto read() { return ReadGuard(g); }
    auto write() { return WriteGuard(g); }

private:
    Guard g;    
}; // struct IOResource


namespace trait
{
    template< typename T >
    struct BuildProperties< ioresource::ReadGuard<T> >
    {
        template <typename Builder>
        static void build( Builder & builder, ioresource::ReadGuard<T> const & g )
        {
            g.build_properties( builder );
        }
    };

    template< typename T >
    struct BuildProperties< ioresource::WriteGuard<T> >
    {
        template <typename Builder>
        static void build( Builder & builder, ioresource::WriteGuard<T> const & g )
        {
            g.build_properties( builder );    
        }
    };
} // namespace trait

} // namespace redGrapes
