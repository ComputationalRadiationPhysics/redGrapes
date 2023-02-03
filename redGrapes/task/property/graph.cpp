/* Copyright 2019-2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <memory>
#include <unordered_set>

#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/property/graph.hpp>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/context.hpp>
#include <redGrapes/resource/resource_user.hpp>

namespace redGrapes
{

/*! create a new (external) event which precedes the tasks post-event
 */
scheduler::EventPtr GraphProperty::make_event()
{
    auto event = memory::alloc_shared< scheduler::Event >();
    event->add_follower( get_post_event() );
    return scheduler::EventPtr{ scheduler::T_EVT_EXT, nullptr, event };
}

/*!
 * Insert a new task and add the same dependencies as in the precedence graph.
 * Note that tasks must be added in order, since only preceding tasks are considered!
 *
 * The precedence graph containing the task is assumed to be locked.
 */
void GraphProperty::init_graph()
{
    SPDLOG_TRACE("sg init task {}", this->task->task_id);

    for( ResourceEntry & r : this->task->unique_resources )
    {
        if( r.task_idx > 0 )
        {
            std::shared_lock< std::shared_mutex > lock( r.resource->users_mutex );
            for(auto it = r.resource->users.iter_from( r.task_idx-1 ); it.first != it.second; ++it.first )
            {
                Task * preceding_task = *it.first;

                if( preceding_task == this->space->parent )
                    break;

                if(
                   preceding_task->space == this->space &&
                   this->space->is_serial( *preceding_task, *this->task )
                )
                {
                    SPDLOG_TRACE("add dependency: task {} -> task {}", preceding_task->task_id, this->task->task_id);
                    add_dependency( *preceding_task );

                    if( preceding_task->has_sync_access( r.resource ) )
                        break;
                }
            }
        }
    }

    // add dependency to parent
    if( auto parent = this->space->parent )
    {
        SPDLOG_TRACE("add event dep to parent");
        this->post_event.add_follower( parent->get_post_event() );
    }
}

void GraphProperty::delete_from_resources()
{
    for( ResourceEntry r : this->task->unique_resources )
    {
        std::unique_lock< std::shared_mutex > lock( r.resource->users_mutex );
        if( r.task_idx != -1 )
            r.resource->users.remove( r.task_idx );
    }
}

void GraphProperty::add_dependency( Task & preceding_task )
{
    // precedence graph
    //in_edges.push_back(&preceding_task);

    // scheduling graph
    auto preceding_event =
        top_scheduler->task_dependency_type(preceding_task, *this->task)
        ? preceding_task->get_pre_event() : preceding_task->get_post_event();
    
    if(! preceding_event->is_reached() )
        preceding_event->add_follower( this->get_pre_event() );
}

void GraphProperty::update_graph( )
{
    std::unique_lock< SpinLock > lock( post_event.followers_mutex );

    for( auto it = post_event.followers.iter(); it.first != it.second; ++it.first )
    {
        auto follower = *it.first;
        if( follower.task )
        {
            if( ! space->is_serial(*this->task, *follower.task) )
            {
                // remove dependency
                //follower.task->in_edges.erase(std::find(std::begin(follower.task->in_edges), std::end(follower.task->in_edges), this));
                post_event.followers.erase(follower);

                follower.notify();
            }
        }
    }
}

} // namespace redGrapes


