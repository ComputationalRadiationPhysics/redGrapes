/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/util/chunked_list.hpp>

#include <spdlog/spdlog.h>

#include <atomic>
#include <cassert>
#include <memory>
#include <mutex>
#include <shared_mutex>

#ifndef REDGRAPES_EVENT_FOLLOWER_LIST_CHUNKSIZE
#    define REDGRAPES_EVENT_FOLLOWER_LIST_CHUNKSIZE 16
#endif

namespace redGrapes
{

    struct Task;

    namespace scheduler
    {

        struct Event;

        enum EventPtrTag
        {
            T_UNINITIALIZED = 0,
            T_EVT_PRE,
            T_EVT_POST,
            T_EVT_RES_SET,
            T_EVT_RES_GET,
            T_EVT_EXT,
        };

        struct EventPtr
        {
            std::shared_ptr<Event> external_event;
            Task* task = nullptr;
            enum EventPtrTag tag = T_UNINITIALIZED;

            inline operator bool() const
            {
                return tag != T_UNINITIALIZED && (task || tag == T_EVT_EXT);
            }

            inline bool operator==(EventPtr const& other) const
            {
                return this->tag == other.tag && this->task == other.task;
            }

            Event& get_event() const;

            inline Event& operator*() const
            {
                return get_event();
            }

            inline Event* operator->() const
            {
                return &get_event();
            }

            /*! A preceding event was reached and thus an incoming edge got removed.
             * This events state is decremented and recursively notifies its followers
             * in case it is now also reached.
             * @return true if event was ready
             */
            bool notify(bool claimed = false);
        };

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
            std::atomic_uint16_t state;

            //! waker that is waiting for this event
            WakerId waker_id;

            //! the set of subsequent events
            ChunkedList<EventPtr, REDGRAPES_EVENT_FOLLOWER_LIST_CHUNKSIZE> followers;

            Event();
            Event(Event&);
            Event(Event&&);

            bool is_reached();
            bool is_ready();
            void up();
            void dn();

            //! note: follower has to be notified separately!
            void remove_follower(EventPtr follower);
            void add_follower(EventPtr follower);

            void notify_followers();
        };

    } // namespace scheduler

} // namespace redGrapes
