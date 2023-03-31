/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <cassert>
#include <memory>
#include <spdlog/spdlog.h>

#include <redGrapes/task/task.hpp>
#include <redGrapes/task/property/id.hpp>
#include <redGrapes/task/task_space.hpp>

#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/context.hpp>
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/util/trace.hpp>

namespace redGrapes
{
namespace scheduler
{

Event::Event()
  : state(1)
{
}

Event::Event(Event & other)
    : state((uint16_t)other.state)
    , waker_id(other.waker_id)
{
}

Event::Event(Event && other)
    : state((uint16_t)other.state)
    , waker_id(other.waker_id)
{
}

bool Event::is_reached() { return state == 0; }
bool Event::is_ready() { return state == 1; }
void Event::up() { state++; }
void Event::dn() { state--; }

void Event::add_follower( EventPtr follower )
{
    TRACE_EVENT("Event", "add_follower");

    std::unique_lock< SpinLock > lock( followers_mutex );
    if( !is_reached() )
    {
        SPDLOG_TRACE("Event add follower");
        followers.push(follower);
        follower->state++;
    }
}

//! note: follower has to be notified separately!
void Event::remove_follower( EventPtr follower )
{
    TRACE_EVENT("Event", "remove_follower");

    std::unique_lock< SpinLock > lock( followers_mutex );
    followers.erase( follower );
}

void Event::notify_followers()
{
    TRACE_EVENT("Event", "notify_followers");

    std::unique_lock< SpinLock > lock( followers_mutex );
    for( auto it = followers.iter(); it.first != it.second; ++it.first )
    {
        if( std::optional<EventPtr> follower = *it.first)
            follower->notify( );
    }
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
bool EventPtr::notify( bool claimed )
{
    TRACE_EVENT("Event", "notify");

    int old_state = this->get_event().state.fetch_sub(1);
    int state = old_state - 1;
    
    std::string tag_string;
    switch( this->tag )
    {
    case EventPtrTag::T_EVT_PRE: tag_string = "pre-event"; break;
    case EventPtrTag::T_EVT_POST: tag_string = "post-event"; break;
    case EventPtrTag::T_EVT_RES_SET: tag_string = "result-set"; break;
    case EventPtrTag::T_EVT_RES_GET: tag_string = "result-get"; break;
    case EventPtrTag::T_EVT_EXT: tag_string = "external"; break;
    }

    if( this->task )
        SPDLOG_TRACE("notify event {} ({}-event of task {}) ~~> state = {}",
               (void *)&this->get_event(), tag_string, this->task->task_id, state);


    assert(old_state > 0);

    bool remove_task = false;

    if(task)
    {
        // pre event ready
        if(tag == scheduler::T_EVT_PRE && state == 1)
        {
            if(!claimed)
                top_scheduler->activate_task(*task);
        }

        // post event reached:
        // no other task can now create dependencies
        if( state == 0 && tag == scheduler::T_EVT_POST )
            task->delete_from_resources();
    }

    // if event is ready or reached (state ∈ {0,1})
    if( state <= 1 && this->get_event().waker_id >= 0 )
        top_scheduler->wake( this->get_event().waker_id );

    if( state == 0 )
    {
        this->get_event().notify_followers();

        // the second one of either post-event or result-get-event shall destroy the task
        if( task )
            if( tag == scheduler::T_EVT_POST
             || tag == scheduler::T_EVT_RES_GET )
                if( task->removal_countdown.fetch_sub(1) == 1 )
                    task->space->free_task( task );
    }

    // return true if event is ready (state == 1)
    return state == 1;
}

} // namespace scheduler

} // namespace redGrapes



