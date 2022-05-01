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

struct EventPtr;

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
struct Event
{
    /*! number of incoming edges
     * state == 0: event is reached and can be removed
     */
    std::atomic_int state;

    //! the set of subsequent events
    std::vector< EventPtr > followers;
    std::shared_mutex followers_mutex;

    Event();
    Event(Event &);
    Event(Event &&);

    bool is_reached();
    bool is_ready();
    void up();
    void dn();

    //! note: follower has to be notified separately!
    void remove_follower( EventPtr follower );
    void add_follower( EventPtr follower );
};



enum EventPtrTag {
    T_UNINITIALIZED = 0,
    T_EVT_PRE,
    T_EVT_POST,
    T_EVT_RES,
    T_EVT_EXT,
};

struct EventPtr
{
    enum EventPtrTag tag;

    std::weak_ptr< PrecedenceGraphVertex > task_vertex;
    std::shared_ptr< Event > external_event;

    bool operator==( EventPtr const & other );

    Event & get_event() const;
    Event & operator*() const;
    Event * operator->() const;
    
    /*! A preceding event was reached and thus an incoming edge got removed.
     * This events state is decremented and recursively notifies its followers
     * in case it is now also reached.
     *
     * @param hook 
     * @return previous state of event
     */
    template < typename F >
    int notify( F && hook );
};

} // namespace scheduler

} // namespace redGrapes


