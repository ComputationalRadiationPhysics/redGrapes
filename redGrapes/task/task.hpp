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

namespace redGrapes
{
namespace task
{

    template<typename... TaskPropertyPolicies>
    struct PropTask : TaskBase, TaskProperties<TaskPropertyPolicies...>
    {
        using Props = TaskProperties<TaskPropertyPolicies...>;
        using VertexPtr = std::shared_ptr< PrecedenceGraphVertex >;
        using WeakVertexPtr = std::weak_ptr< PrecedenceGraphVertex >;

        unsigned int scope_level;

        virtual ~PropTask() {}

        PropTask(TaskProperties<TaskPropertyPolicies...>&& prop)
            : TaskProperties<TaskPropertyPolicies...>(prop)
        {
        }
    };

    template<typename F, typename... TaskPropertyPolicies>
    struct FunTask : PropTask<TaskPropertyPolicies...>
    {
        F impl;

        virtual ~FunTask() {}

        FunTask(F&& f, TaskProperties<TaskPropertyPolicies...>&& prop)
            : PropTask<TaskPropertyPolicies...>(std::move(prop))
            , impl(std::move(f))
        {
        }

        void run()
        {
            impl();
        }
    };


template <typename F, typename... TaskPropertyPolicies>
std::unique_ptr<FunTask<F, TaskPropertyPolicies...>> make_fun_task(F&& f, TaskProperties<TaskPropertyPolicies...> prop) {
    return std::make_unique<FunTask<F, TaskPropertyPolicies...>>(std::move(f), std::move(prop));
}


} // namespace task

} // namespace redGrapes

