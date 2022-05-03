/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <redGrapes/task/task_base.hpp>
#include <redGrapes/task/property/inherit.hpp>
#include <redGrapes/task/property/trait.hpp>
#include <redGrapes/task/property/id.hpp>
#include <redGrapes/task/property/resource.hpp>
#include <redGrapes/task/property/graph.hpp>

namespace redGrapes
{

using TaskProperties = TaskProperties1<
    IDProperty,
    ResourceProperty,
    GraphProperty
#ifdef REDGRAPES_TASK_PROPERTIES
    , REDGRAPES_TASK_PROPERTIES
#endif
>;

struct Task :
        TaskBase,
        TaskProperties,
        std::enable_shared_from_this<Task>
{
    virtual ~Task() {}
    
    Task(TaskProperties && prop)
        : TaskProperties(std::move(prop))
    {
    }

};

template<typename F>
struct FunTask : Task
{
    F impl;

    virtual ~FunTask() {}

    FunTask(F&& f, TaskProperties&& prop)
        : Task(std::move(prop))
        , impl(std::move(f))
    {
    }

    void run()
    {
        impl( *this );
    }
};

template<typename F>
std::shared_ptr<FunTask<F>> make_fun_task(F&& f, TaskProperties && prop)
{
    return std::make_shared<FunTask<F>>(std::move(f), std::move(prop));
}

} // namespace redGrapes

