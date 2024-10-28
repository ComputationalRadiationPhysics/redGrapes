/* Copyright 2022-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/dispatch/thread/DefaultWorker.hpp"
#include "redGrapes/dispatch/thread/WorkerThread.hpp"
#include "redGrapes/dispatch/thread/worker_pool.hpp"
#include "redGrapes/memory/allocator.hpp"
#include "redGrapes/memory/chunked_bump_alloc.hpp"
#include "redGrapes/memory/hwloc_alloc.hpp"
#include "redGrapes/util/trace.hpp"

#include <memory>

namespace redGrapes
{
    namespace dispatch
    {
        namespace thread
        {
            template<typename Worker>
            WorkerPool<Worker>::WorkerPool(size_t n_workers) : worker_state(n_workers)
                                                             , num_workers(n_workers)
            {
            }

            template<typename Worker>
            void WorkerPool<Worker>::emplace_workers(WorkerId base_id)
            {
                m_base_id = base_id;
                if(num_workers > TaskFreeCtx::n_pus)
                    SPDLOG_WARN(
                        "{} worker-threads requested, but only {} PUs available!",
                        num_workers,
                        TaskFreeCtx::n_pus);

                workers.reserve(num_workers);

                SPDLOG_DEBUG("populate WorkerPool with {} workers", num_workers);
                for(WorkerId worker_id = base_id; worker_id < base_id + num_workers; ++worker_id)
                {
                    WorkerId pu_id = worker_id % TaskFreeCtx::n_pus;
                    // allocate worker with id `i` on arena `i`,
                    hwloc_obj_t obj = hwloc_get_obj_by_type(TaskFreeCtx::hwloc_ctx.topology, HWLOC_OBJ_PU, pu_id);
                    TaskFreeCtx::worker_alloc_pool.allocs.push_back(
                        std::make_unique<memory::ChunkedBumpAlloc<memory::HwlocAlloc>>(
                            memory::HwlocAlloc(TaskFreeCtx::hwloc_ctx, obj),
                            REDGRAPES_ALLOC_CHUNKSIZE));

                    auto worker = memory::alloc_shared_bind<WorkerThread<Worker>>(worker_id, obj, worker_id, *this);
                    workers.emplace_back(worker);
                }
            }

            template<typename Worker>
            void WorkerPool<Worker>::start()
            {
                for(auto& worker : workers)
                    worker->start();
            }

            template<typename Worker>
            void WorkerPool<Worker>::stop()
            {
                for(auto& worker : workers)
                    worker->stop();

                workers.clear();
            }

            template<typename Worker>
            int WorkerPool<Worker>::find_free_worker()
            {
                TRACE_EVENT("Scheduler", "find_worker");

                SPDLOG_TRACE("find worker...");

                WorkerId start_idx = 0;

                if(m_base_id <= *TaskFreeCtx::current_worker_id
                   && *TaskFreeCtx::current_worker_id < m_base_id + num_workers)
                    start_idx = *TaskFreeCtx::current_worker_id - m_base_id;

                std::optional<WorkerId> idx = this->probe_worker_by_state<WorkerId>(
                    [this](WorkerId idx) -> std::optional<WorkerId>
                    {
                        if(set_worker_state(idx, WorkerState::BUSY))
                            return idx;
                        else
                            return std::nullopt;
                    },
                    dispatch::thread::WorkerState::AVAILABLE, // find a free worker
                    start_idx,
                    false);

                if(idx)
                    return *idx;
                else
                    // no free worker found,
                    return -1;
            }

        } // namespace thread
    } // namespace dispatch
} // namespace redGrapes
