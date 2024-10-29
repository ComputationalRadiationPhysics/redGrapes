/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/resource/ioresource.hpp
 */

#pragma once

#include "redGrapes/resource/access/io.hpp"
#include "redGrapes/resource/resource.hpp"

#include <memory>
#include <utility>

namespace redGrapes
{

    template<typename T>
    struct IOResource : public SharedResourceObject<T, access::IOAccess>
    {
        using SharedResourceObject<T, access::IOAccess>::SharedResourceObject;

        auto read() const noexcept
        {
            return ResourceAccessPair<std::shared_ptr<T const>>{
                this->obj,
                this->res.make_access(access::IOAccess::read)};
        }

        auto write() const noexcept
        {
            return ResourceAccessPair<std::shared_ptr<T>>{this->obj, this->res.make_access(access::IOAccess::write)};
        }
    }; // struct IOResource

    template<typename T>
    IOResource(std::shared_ptr<T>) -> IOResource<T>;

} // namespace redGrapes
