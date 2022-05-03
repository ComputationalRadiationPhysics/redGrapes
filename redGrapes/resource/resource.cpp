/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/context.hpp>
#include <redGrapes/resource/resource.hpp>

namespace redGrapes
{

int ResourceBase::getID()
{
    static std::atomic_int id_counter;
    return id_counter.fetch_add(1);
}

ResourceBase::ResourceBase()
    : id( getID() )
    , scope_level( scope_depth() )
{}

} // namespace redGrapes

