/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/scheduler/scheduling_graph.hpp>

namespace redGrapes
{
namespace scheduler
{

bool EventPtr::operator==( EventPtr const & other )
{
    return this->tag == other.tag && this->task_vertex.lock() == other.task_vertex.lock();
}

Event & EventPtr::get_event() const
{
    switch( tag )
    {
    case T_EVT_PRE:
        return task_vertex.lock()->template get_dyn_task<SchedulingGraphProp>().pre_event;
    case T_EVT_POST:
        return task_vertex.lock()->template get_dyn_task<SchedulingGraphProp>().post_event;
    case T_EVT_RES:
        return task_vertex.lock()->template get_dyn_task<SchedulingGraphProp>().result_event;
    case T_EVT_EXT:
        return *external_event;
    default:
        throw std::runtime_error("invalid event tag");
    }    
}

Event & EventPtr::operator*() const
{
    return get_event();
}

Event * EventPtr::operator->() const
{
    return &get_event();
}

} // namespace scheduler

} // namespace redGrapes

