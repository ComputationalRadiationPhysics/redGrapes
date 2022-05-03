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
    : state((int)other.state)
{
}

Event::Event(Event && other)
    : state((int)other.state)
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
 * @param hook 
 * @return previous state of event
 */
void EventPtr::notify()
{
    int old_state = this->get_event().state.fetch_sub(1);

    assert( old_state > 0 );

    if( task )
    {
        // pre event ready
        if( tag == scheduler::T_EVT_PRE && (old_state-1) == 1 )
        {
            SPDLOG_TRACE("pre event ready");
            top_scheduler->activate_task(task);
        }

        // post event reached
        if( tag == scheduler::T_EVT_POST && (old_state-1) == 0 )
        {
            SPDLOG_TRACE("post event reached");
            if(auto children = task->children)
                while(auto new_task = children->next())
                {
                    new_task->sg_init();
                    new_task->pre_event.up();
                    new_task->get_pre_event().notify();
                }
            else
                remove_task(task);
        }
    }

    if( old_state == 1 )
    {
        // notify followers
        std::shared_lock< std::shared_mutex > lock( this->get_event().followers_mutex );
        for( auto & follower : this->get_event().followers )
            follower.notify();
    }

    if(top_scheduler)
        top_scheduler->notify();
}

} // namespace scheduler

} // namespace redGrapes



