/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <mutex>
#include <redGrapes/context.hpp>
#include <redGrapes/resource/resource.hpp>

namespace redGrapes
{
struct Task;

unsigned int ResourceBase::generateID()
{
    static std::atomic< unsigned int > id_counter;
    return id_counter.fetch_add(1);
}

ResourceBase::ResourceBase()
    : id( generateID() )
    , scope_level( scope_depth() )
    , users(
        memory::Allocator< uint8_t >( get_arena_id() ),
        REDGRAPES_RUL_CHUNKSIZE
    )
{}

} // namespace redGrapes

