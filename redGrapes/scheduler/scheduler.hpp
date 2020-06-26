/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace redGrapes
{
namespace scheduler
{

template <
    typename TaskPtr
>
struct IScheduler
{
    virtual ~IScheduler() {}

    virtual bool task_dependency_type( TaskPtr a, TaskPtr b )
    {
        return false;
    }

    virtual void activate_task( TaskPtr task_ptr ) = 0;
};

} // namespace scheduler

} // namespace redGrapes

