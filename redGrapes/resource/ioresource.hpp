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

namespace redGrapes
{

template < typename T >
struct IOResource
{
    struct Guard
    {
    protected:
        friend class IOResource;

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

    struct ReadGuard : Guard
    {
        ReadGuard read() const noexcept { return *this; }
        T const & operator* () const noexcept { return *this->obj; }
        T const * operator-> () const noexcept { return this->obj.get(); }

        template <typename Builder>
        void build_properties( Builder & builder )
        {
            builder.add_resource( this->resource.make_access( access::IOAccess::read ) );
        }

    private:
        friend class IOResource;
        ReadGuard(Guard const & other) : Guard(other) {}
    };

    struct WriteGuard : ReadGuard
    {
        WriteGuard write() const noexcept { return *this; }
        T & operator* () const noexcept { return *this->obj; }
        T * operator-> () const noexcept { return this->obj.get(); }

        template <typename B>
        void build_properties( ResourceProperty::Builder<B> & builder )
        {
            builder.add_resource( this->resource.make_access( access::IOAccess::write ) );
        }

    private:
        friend class IOResource;
        WriteGuard(Guard const & other) : ReadGuard(other) {}
    };

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

} // namespace redGrapes
