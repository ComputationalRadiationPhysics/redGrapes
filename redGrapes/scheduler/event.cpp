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

#include <redGrapes/task/itask.hpp>
#include <redGrapes/task/property/id.hpp>
#include <redGrapes/task/task_space.hpp>

#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/context.hpp>
#include <redGrapes/redGrapes.hpp>

namespace redGrapes
{
namespace scheduler
{

Event::Event()
  : state(1)
{
}

Event::Event(Event & other)
  : state((int)other.state), waker(other.waker)
{
}

Event::Event(Event && other)
  : state((int)other.state), waker(other.waker)
{    
}

bool Event::is_reached() { return state == 0; }
bool Event::is_ready() { return state == 1; }
void Event::up() { state++; }
void Event::dn() { state--; }

void Event::add_follower( EventPtr follower )
{
    std::unique_lock< std::shared_mutex > lock( followers_mutex );

    if( !is_reached() )
    {
        followers.push_back(follower);
        follower->state++;
    }
}

//! note: follower has to be notified separately!
void Event::remove_follower( EventPtr follower )
{
    //SPDLOG_TRACE("event {} remove_follower {}", (void*)this, (void*)follower.get());
    std::unique_lock< std::shared_mutex > lock( followers_mutex );
    followers.erase(std::find( std::begin(followers), std::end(followers), follower ));
}

/*! A preceding event was reached and thus an incoming edge got removed.
 * This events state is decremented and recursively notifies its followers
 * in case it is now also reached.
 *
 * @return true if event is ready
 */
bool EventPtr::notify( )
{
    SPDLOG_TRACE("notify event {}", (void*)&this->get_event() );
    int old_state = this->get_event().state.fetch_sub(1);

    assert( old_state > 0 );

    bool remove_task = false;

    if( task )
    {
        // pre event ready
        if( tag == scheduler::T_EVT_PRE && old_state == 2 )
        {
            top_scheduler->activate_task(*task);
            schedule();
        }

        // post event or result-get event reached
        if(
           old_state == 1 &&
           (tag == scheduler::T_EVT_POST ||
            tag == scheduler::T_EVT_RES_GET)
        )
        {
            if(auto children = task->children)
                children->init_until_ready();

            remove_task = true;
        }
    }

    if( old_state == 1 )
    {
        // notify followers
        std::shared_lock< std::shared_mutex > lock( this->get_event().followers_mutex );
        for( auto & follower : this->get_event().followers )
        {
            follower.notify( );
        }
    }

    if( old_state <= 2 )
    {
	if(auto w = this->get_event().waker.lock())
        {
	    this->get_event().waker.reset();
            w->wake();
        }
    }

    if( remove_task )
        task->space->try_remove(*task);

    return old_state == 2;
}

} // namespace scheduler

} // namespace redGrapes



