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
#include <redGrapes/resource/resource_user.hpp>
#include <redGrapes/util/trace.hpp>
#include <redGrapes/redGrapes.hpp>

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
    TRACE_EVENT("Graph", "init_graph");
    for( auto r = this->task->unique_resources.rbegin(); r != this->task->unique_resources.rend(); ++r )
    {
        if( r->task_entry != r->resource->users.rend() )
        {
            // TODO: can this lock be avoided?
            //
            //   even though the container supports
            //   lock free iteration and removal,
            //   with out this lock, its still possible,
            //   that the iterator points at an element,
            //   which will get removed AFTER iterating
            //   and BEFORE adding the dependency.
            std::unique_lock< SpinLock > lock( r->resource->users_mutex );

            TRACE_EVENT("Graph", "CheckPredecessors");
            auto it = r->task_entry;

            ++it;
            for(; it != r->resource->users.rend(); ++it )
            {
                TRACE_EVENT("Graph", "Check Pred");
                Task * preceding_task = *it;

                if( preceding_task == this->space->parent )
                    break;

                if(
                   preceding_task->space == this->space &&
                   this->space->is_serial( *preceding_task, *this->task )
                )
                {
                    add_dependency( *preceding_task );
                    if( preceding_task->has_sync_access( r->resource ) )
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
    TRACE_EVENT("Graph", "delete_from_resources");
    for( auto r = this->task->unique_resources.rbegin(); r != this->task->unique_resources.rend(); ++r )
    {
        // TODO: can this lock be avoided?
        //   corresponding lock to init_graph()
        std::unique_lock< SpinLock > lock( r->resource->users_mutex );

        if( r->task_entry != r->resource->users.rend() )
            r->resource->users.remove( r->task_entry );
    }
}

void GraphProperty::add_dependency( Task & preceding_task )
{
    // precedence graph
    //in_edges.push_back(&preceding_task);

    // scheduling graph
    auto preceding_event =
        SingletonContext::get().scheduler->task_dependency_type(preceding_task, *this->task)
        ? preceding_task->get_pre_event() : preceding_task->get_post_event();

    if( ! preceding_event->is_reached() )
        preceding_event->add_follower( this->get_pre_event() );
}

void GraphProperty::update_graph( )
{
    //std::unique_lock< SpinLock > lock( post_event.followers_mutex );

    //    for( auto follower : post_event.followers )
    for( auto it = post_event.followers.rbegin(); it != post_event.followers.rend(); ++it )
    {
        scheduler::EventPtr follower = *it;
        if( follower.task )
        {
            if( ! space->is_serial(*this->task, *follower.task) )
            {
                // remove dependency
                //follower.task->in_edges.erase(std::find(std::begin(follower.task->in_edges), std::end(follower.task->in_edges), this));
                post_event.followers.erase( follower );

                follower.notify();
            }
        }
    }
}

} // namespace redGrapes


