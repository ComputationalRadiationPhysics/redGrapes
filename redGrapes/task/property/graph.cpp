/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <memory>
#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/property/graph.hpp>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/context.hpp>

namespace redGrapes
{

GraphProperty::GraphProperty() {}
GraphProperty::GraphProperty(GraphProperty const & other)
    : space(other.space)
    , scope_depth(other.scope_depth)
    , xty_task(other.xty_task)
{}

std::shared_ptr<Task> GraphProperty::get_task() {
    return xty_task.lock();
}

bool GraphProperty::is_ready() { return pre_event.is_ready(); }
bool GraphProperty::is_running() { return pre_event.is_reached(); }
bool GraphProperty::is_finished() { return post_event.is_reached(); }

void GraphProperty::add_dependency( std::shared_ptr<Task> preceding_task )
{
    in_edges.push_back(preceding_task);
}

scheduler::EventPtr GraphProperty::get_pre_event()
{
    return scheduler::EventPtr { scheduler::T_EVT_PRE, this->get_task() };
}

scheduler::EventPtr GraphProperty::get_post_event()
{
    return scheduler::EventPtr { scheduler::T_EVT_POST, this->get_task() };
}

scheduler::EventPtr GraphProperty::get_result_event()
{
    return scheduler::EventPtr { scheduler::T_EVT_RES, this->get_task() };
}

/*! create a new (external) event which precedes the tasks post-event
 */
scheduler::EventPtr GraphProperty::make_event()
{
    auto event = std::make_shared< scheduler::Event >();
    event->add_follower( get_post_event() );
    return scheduler::EventPtr{ scheduler::T_EVT_EXT, std::shared_ptr<Task>(), event };
}

/*!
 * represent ›pausation of the task until event is reached‹
 * in the scheduling graph
 */
void GraphProperty::sg_pause( scheduler::EventPtr event )
{
    //SPDLOG_TRACE("sg pause: new_event = {}", (void*) event.get());
    pre_event.state = 1;
    event->add_follower( get_pre_event() );
}

/*!
 * Insert a new task and add the same dependencies as in the precedence graph.
 * Note that tasks must be added in order, since only preceding tasks are considered!
 *
 * The precedence graph containing the task is assumed to be locked.
 */
void GraphProperty::sg_init()
{
    SPDLOG_TRACE("sg init task {}", get_task()->task_id);

    
    
    // add dependencies to tasks which precede the new one
    for(auto weak_in_vertex : get_task()->in_edges)
    {
        if( auto preceding_task = weak_in_vertex.lock() )
        {
            auto preceding_event =
                top_scheduler->task_dependency_type(preceding_task, get_task())
                ? preceding_task->get_pre_event() : preceding_task->get_post_event();

            if(! preceding_event->is_reached() )
                preceding_event->add_follower( this->get_pre_event() );
        }
    }

    // add dependency to parent
    if( auto parent = get_task()->space->parent.lock() )
        parent->post_event.add_follower( this->get_post_event() );
}

} // namespace redGrapes


