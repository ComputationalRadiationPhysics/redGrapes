/* Copyright 2022-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "redGrapes/scheduler/event.hpp"

namespace redGrapes
{
    namespace scheduler
    {
        template<typename TTask>
        Event<TTask>& EventPtr<TTask>::get_event() const
        {
            switch(tag)
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

    } // namespace scheduler

} // namespace redGrapes
