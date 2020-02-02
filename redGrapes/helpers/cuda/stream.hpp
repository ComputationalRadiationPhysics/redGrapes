/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <redGrapes/resource/resource.hpp>
#include <redGrapes/property/trait.hpp>
#include <redGrapes/helpers/cuda/synchronize.hpp>

namespace redGrapes
{
namespace helpers
{
namespace cuda
{

struct StreamAccess
{
    static bool is_serial(StreamAccess const & a, StreamAccess const & b)
    {
        return true;
    }

    bool is_superset_of(StreamAccess const & sub) const
    {
        return true;
    }

    friend bool operator==(StreamAccess const &, StreamAccess const &)
    {
        return true;
    }

    friend std::ostream& operator<<(std::ostream& out, StreamAccess const & a)
    {
        out << "Stream Access" << std::endl;
        return out;
    }
};

template< typename Manager >
struct StreamResource
    : SharedResourceObject< StreamSynchronizer< Manager >, StreamAccess >
{
    operator ResourceAccess() const noexcept
    {
        return this->make_access( StreamAccess{} );
    }

    operator cudaStream_t() const noexcept
    {
        return this->obj->cuda_stream;
    }

    void poll() const
    {
        this->obj->poll();
    }

    void sync() const
    {
        this->obj->sync();
    }

    StreamResource( Manager & mgr, cudaStream_t cudaStream )
        : SharedResourceObject<StreamSynchronizer<Manager>, StreamAccess>( std::make_shared<StreamSynchronizer<Manager>>( mgr, cudaStream ) )
    {}
};

} // namespace cuda

} // namespace helpers

} // namespace redGrapes

