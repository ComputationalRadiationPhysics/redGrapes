/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/TaskCtx.hpp"
#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/redGrapes.hpp"
#include "redGrapes/util/trace.hpp"

#include <moodycamel/concurrentqueue.h>

#include <functional>

#if REDGRAPES_ENABLE_TRACE
PERFETTO_TRACK_EVENT_STATIC_STORAGE();
#endif

namespace redGrapes
{

    //! get backtrace from currently running task
    template<typename TSchedMap, C_TaskProperty... TUserTaskProperties>
    std::vector<std::reference_wrapper<Task<TUserTaskProperties...>>> RedGrapes<TSchedMap, TUserTaskProperties...>::
        backtrace() const
    {
        std::vector<std::reference_wrapper<RGTask>> bt;
        for(RGTask* task = TaskCtx<RGTask>::current_task; task != nullptr; task = task->space->parent)
            bt.push_back(*task);

        return bt;
    }

    template<typename TSchedMap, C_TaskProperty... TUserTaskProperties>
    void RedGrapes<TSchedMap, TUserTaskProperties...>::init_tracing()
    {
#if REDGRAPES_ENABLE_TRACE
        perfetto::TracingInitArgs args;
        args.backends |= perfetto::kInProcessBackend;
        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();

        tracing_session = StartTracing();
#endif
    }

    template<typename TSchedMap, C_TaskProperty... TUserTaskProperties>
    void RedGrapes<TSchedMap, TUserTaskProperties...>::finalize_tracing()
    {
#if REDGRAPES_ENABLE_TRACE
        StopTracing(tracing_session);
#endif
    }

    /*! wait until all tasks in the current task space finished
     */
    template<typename TSchedMap, C_TaskProperty... TUserTaskProperties>
    void RedGrapes<TSchedMap, TUserTaskProperties...>::barrier()
    {
        SPDLOG_TRACE("barrier");

        while(!TaskCtx<RGTask>::root_space->empty())
            TaskFreeCtx::idle();
    }

    //! apply a patch to the properties of the currently running task
    template<typename TSchedMap, C_TaskProperty... TUserTaskProperties>
    void RedGrapes<TSchedMap, TUserTaskProperties...>::update_properties(
        typename Task<TUserTaskProperties...>::TaskProperties::Patch const& patch)
    {
        if(TaskCtx<RGTask>::current_task)
        {
            TaskCtx<RGTask>::current_task->apply_patch(patch);
            TaskCtx<RGTask>::current_task->update_graph();
        }
        else
            throw std::runtime_error("update_properties: currently no task running");
    }

} // namespace redGrapes
