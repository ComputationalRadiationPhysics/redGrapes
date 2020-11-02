/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <vector>
#include <mutex>

namespace redGrapes
{
namespace helpers
{
namespace cupla
{

//! Manages the recycling of cuda events
struct EventPool
{
public:
    static auto & get()
    {
        static EventPool singleton;
        return singleton;
    }

    ~EventPool()
    {
        std::lock_guard< std::mutex > lock( mutex );
        for( auto e : unused_cupla_events )
            cuplaEventDestroy( e );
    }

    cuplaEvent_t alloc()
    {
        std::lock_guard< std::mutex > lock( mutex );

        cuplaEvent_t e;

        if( unused_cupla_events.empty() )
            cuplaEventCreate( &e );
        else
        {
            e = unused_cupla_events.back();
            unused_cupla_events.pop_back();
        }

        return e;
    }

    void free( cuplaEvent_t event )
    {
        std::lock_guard< std::mutex > lock( mutex );
        unused_cupla_events.push_back( event );
    }

private:
    std::mutex mutex;
    std::vector< cuplaEvent_t > unused_cupla_events;

};

} // namespace cupla

} // namespace helpers
    
} // namespace redGrapes

