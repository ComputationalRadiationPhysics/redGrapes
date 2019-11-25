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

struct IOResource : public Resource< access::IOAccess >
{
#define OP(name)                                                                \
    inline ResourceAccess name (void) const                                     \
    { return this->make_access(access::IOAccess{access::IOAccess::name}); }

    OP(read)
    OP(write)
    OP(aadd)
    OP(amul)

#undef OP
}; // struct IOResource

} // namespace redGrapes
