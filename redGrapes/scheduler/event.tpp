/* Copyright 2022-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "redGrapes/scheduler/event.hpp"
#include "redGrapes/util/trace.hpp"

#include <spdlog/spdlog.h>

#include <atomic>
#include <cassert>

namespace redGrapes
{
    namespace scheduler
    {
        template<typename TTask>
        Event<TTask>::Event() : state(1)
                              , waker_id(-1)
                              , followers(memory::Allocator())
        {
        }

        template<typename TTask>
        Event<TTask>::Event(Event& other)
            : state((uint16_t) other.state)
            , waker_id(other.waker_id)
            , followers(memory::Allocator())
        {
        }

        template<typename TTask>
        Event<TTask>::Event(Event&& other)
            : state((uint16_t) other.state)
            , waker_id(other.waker_id)
            , followers(memory::Allocator())
        {
        }

        template<typename TTask>
        bool Event<TTask>::is_reached()
        {
            return state == 0;
        }

        template<typename TTask>
        bool Event<TTask>::is_ready()
        {
            return state == 1;
        }

        template<typename TTask>
        void Event<TTask>::up()
        {
            state++;
        }

        template<typename TTask>
        void Event<TTask>::dn()
        {
            state--;
        }

        template<typename TTask>
        void Event<TTask>::add_follower(EventPtr<TTask> follower)
        {
            TRACE_EVENT("Event", "add_follower");

            if(!is_reached())
            {
                SPDLOG_TRACE("Event add follower");
                followers.push(follower);
                follower->state++;
            }
        }

        //! note: follower has to be notified separately!
        template<typename TTask>
        void Event<TTask>::remove_follower(EventPtr<TTask> follower)
        {
            TRACE_EVENT("Event", "remove_follower");

            followers.erase(follower);
        }

        template<typename TTask>
        void Event<TTask>::notify_followers()
        {
            TRACE_EVENT("Event", "notify_followers");

            for(auto follower = followers.rbegin(); follower != followers.rend(); ++follower)
                follower->notify();
        }

        /*! A preceding event was reached and thus an incoming edge got removed.
         * This events state is decremented and recursively notifies its followers
         * in case it is now also reached.
         *
         * @param claimed if true, the scheduler already knows about the task,
         *                if false, activate task is called
         *
         * @return true if event is ready
         */
        template<typename TTask>
        bool EventPtr<TTask>::notify(bool claimed)
        {
            TRACE_EVENT("Event", "notify");

            int old_state = this->get_event().state.fetch_sub(1);
            int state = old_state - 1;

            std::string tag_string;
            switch(this->tag)
            {
            case EventPtrTag::T_EVT_PRE:
                tag_string = "pre";
                break;
            case EventPtrTag::T_EVT_POST:
                tag_string = "post";
                break;
            case EventPtrTag::T_EVT_RES_SET:
                tag_string = "result-set";
                break;
            case EventPtrTag::T_EVT_RES_GET:
                tag_string = "result-get";
                break;
            case EventPtrTag::T_EVT_EXT:
                tag_string = "external";
                break;
            case EventPtrTag::T_UNINITIALIZED:
                tag_string = "uninitialized";
                break;
            }

            if(task)
                SPDLOG_TRACE(
                    "notify event {} ({}-event of task {}) ~~> state = {}",
                    (void*) &get_event(),
                    tag_string,
                    task->task_id,
                    state);

            assert(old_state > 0);

            if(task)
            {
                // pre event ready
                if(tag == scheduler::T_EVT_PRE && state == 1)
                {
                    if(!claimed)
                        task->scheduler.activate_task(*task);
                }

                // post event reached:
                // no other task can now create dependencies to this
                // task after deleting it from the resource list
                if(state == 0 && tag == scheduler::T_EVT_POST)
                    task->delete_from_resources();
            }
            // TODO rework this to reduce if(task) checks
            // if event is ready or reached (state ∈ {0,1})
            if(state <= 1 && get_event().waker_id >= 0)
                task->scheduler.wake(get_event().waker_id);

            if(state == 0)
            {
                get_event().notify_followers();

                // the second one of either post-event or result-get-event shall destroy the task
                if(task)
                    if(tag == scheduler::T_EVT_POST || tag == scheduler::T_EVT_RES_GET)
                    {
                        if(task->removal_countdown.fetch_sub(1) == 1)
                            task->space->free_task(task);
                    }
            }

            // return true if event is ready (state == 1)
            return state == 1;
        }

    } // namespace scheduler

} // namespace redGrapes
