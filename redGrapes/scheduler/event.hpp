/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <cassert>
#include <memory>
#include <spdlog/spdlog.h>

#include <redGrapes/task/itask.hpp>
#include <redGrapes/task/property/id.hpp>
#include <redGrapes/task/task_space.hpp>

namespace std
{
    using shared_mutex = shared_timed_mutex;
} // namespace std

namespace redGrapes
{
namespace scheduler
{

/*!
 * An event is the abstraction of the programs execution state.
 * They form a flat/non-recursive graph of events.
 * During runtime, each thread encounters a sequence of events.
 * The goal is to synchronize these events in the manner
 * "Event A must occur before Event B".
 *
 * Multiple events need to be related, so that they
 * form a partial order.
 * This order is an homomorphic image from the timeline of
 * execution states.
 */
struct Event : std::enable_shared_from_this< Event >
{
    /*! number of incoming edges
     * state == 0: event is reached and can be removed
     */
    std::atomic_int state;

    //! the set of subsequent events
    std::vector< std::shared_ptr< Event > > followers;
    std::shared_mutex followers_mutex;

    std::weak_ptr< PrecedenceGraphVertex > task_vertex;

    Event()
        : state(1)
    {}

    Event( std::weak_ptr< PrecedenceGraphVertex > task_vertex )
        : state(1)
        , task_vertex(task_vertex)
    {}

    bool is_reached() { return state == 0; }
    bool is_ready() { return state == 1; }
    void up() { state++; }
    void dn() { state--; }

    void add_follower( std::shared_ptr<Event> follower )
    {
        SPDLOG_TRACE("event {} add_follower {}", (void*)this, (void*)follower.get());

        std::unique_lock< std::shared_mutex > lock( followers_mutex );

        if( !is_reached() )
        {
            followers.push_back(follower);
            follower->state++;
        }
    }

    //! note: follower has to be notified separately!
    void remove_follower( std::shared_ptr<Event> follower )
    {
        SPDLOG_TRACE("event {} remove_follower {}", (void*)this, (void*)follower.get());

        std::unique_lock< std::shared_mutex > lock( followers_mutex );
        followers.erase(
            std::find( std::begin(followers), std::end(followers), follower )
        );
    }

    /*! A preceding event was reached and thus an incoming edge got removed.
     * This events state is decremented and recursively notifies its followers
     * in case it is now also reached.
     *
     * @param hook 
     * @return previous state of event
     */
    template < typename F >
    int notify( F && hook )
    {
        int old_state = state.fetch_sub(1);

        assert( old_state > 0 );

        hook( old_state - 1, shared_from_this() );

        if( old_state == 1 )
        {
            // notify followers
            std::shared_lock< std::shared_mutex > lock( followers_mutex );
            for( auto & follower : followers )
                follower->notify( hook );
        }

        return state;
    }
};

using EventPtr = std::shared_ptr< scheduler::Event >;

} // namespace scheduler

} // namespace redGrapes


