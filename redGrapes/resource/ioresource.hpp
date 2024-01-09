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

#include <redGrapes/resource/access/io.hpp>
#include <redGrapes/resource/resource.hpp>
#include <redGrapes/task/property/resource.hpp>
#include <redGrapes/task/property/trait.hpp>

namespace redGrapes
{
    namespace ioresource
    {

        template<typename T>
        struct ReadGuard : public SharedResourceObject<T, access::IOAccess>
        {
            operator ResourceAccess() const noexcept
            {
                return this->make_access(access::IOAccess::read);
            }

            ReadGuard read() const noexcept
            {
                return *this;
            }

            T const& operator*() const noexcept
            {
                return *this->obj;
            }

            T const* operator->() const noexcept
            {
                return this->obj.get();
            }

            T const* get() const noexcept
            {
                return this->obj.get();
            }

        protected:
            ReadGuard(std::shared_ptr<T> obj) : SharedResourceObject<T, access::IOAccess>(obj)
            {
            }
        };

        template<typename T>
        struct WriteGuard : public ReadGuard<T>
        {
            operator ResourceAccess() const noexcept
            {
                return this->make_access(access::IOAccess::write);
            }

            WriteGuard write() const noexcept
            {
                return *this;
            }

            T& operator*() const noexcept
            {
                return *this->obj;
            }

            T* operator->() const noexcept
            {
                return this->obj.get();
            }

            T* get() const noexcept
            {
                return this->obj.get();
            }

        protected:
            WriteGuard(std::shared_ptr<T> obj) : ReadGuard<T>(obj)
            {
            }
        };

    } // namespace ioresource

    template<typename T>
    struct IOResource : public ioresource::WriteGuard<T>
    {
        template<typename... Args>
        IOResource(Args&&... args) : ioresource::WriteGuard<T>(memory::alloc_shared<T>(std::forward<Args>(args)...))
        {
        }

        IOResource(std::shared_ptr<T> o) : ioresource::WriteGuard<T>(o)
        {
        }

    }; // struct IOResource

} // namespace redGrapes
