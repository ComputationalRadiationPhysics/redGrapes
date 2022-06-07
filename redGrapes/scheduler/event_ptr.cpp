/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <optional>

#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/property/graph.hpp>
#include <redGrapes/task/task.hpp>

namespace redGrapes
{
namespace scheduler
{

bool EventPtr::operator==( EventPtr const & other ) const
{
    return this->tag == other.tag && this->task == other.task;
}

Event & EventPtr::get_event() const
{    
    switch( tag )
    {
    case T_EVT_PRE:
        return task->pre_event;
    case T_EVT_POST:
        return task->post_event;
    case T_EVT_RES_SET:
        return task->result_set_event;
    case T_EVT_RES_GET:
        return task->result_get_event;
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

