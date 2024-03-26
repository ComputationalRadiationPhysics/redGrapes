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

namespace redGrapes
{
    namespace dispatch
    {
        namespace thread
        {
            template<typename TTask, typename Worker>
            WorkerPool<TTask, Worker>::WorkerPool(HwlocContext& hwloc_ctx, size_t n_workers)
                : hwloc_ctx(hwloc_ctx)
                , worker_state(n_workers)
                , num_workers(n_workers)
            {
            }

            template<typename TTask, typename Worker>
            void WorkerPool<TTask, Worker>::emplace_workers(WorkerId base_id)
            {
                m_base_id = base_id;
                if(num_workers > TaskFreeCtx::n_pus)
                    spdlog::warn(
                        "{} worker-threads requested, but only {} PUs available!",
                        num_workers,
                        TaskFreeCtx::n_pus);

                workers.reserve(num_workers);

                spdlog::debug("populate WorkerPool with {} workers", num_workers);
                for(size_t worker_id = base_id; worker_id < base_id + num_workers; ++worker_id)
                {
                    unsigned pu_id = worker_id % TaskFreeCtx::n_pus;
                    // allocate worker with id `i` on arena `i`,
                    hwloc_obj_t obj = hwloc_get_obj_by_type(TaskFreeCtx::hwloc_ctx.topology, HWLOC_OBJ_PU, pu_id);
                    TaskFreeCtx::worker_alloc_pool->allocs.emplace_back(
                        memory::HwlocAlloc(TaskFreeCtx::hwloc_ctx, obj),
                        REDGRAPES_ALLOC_CHUNKSIZE);

                    auto worker = memory::alloc_shared_bind<WorkerThread<TTask, Worker>>(
                        worker_id,
                        TaskFreeCtx::worker_alloc_pool->get_alloc(worker_id),
                        TaskFreeCtx::hwloc_ctx,
                        obj,
                        worker_id,
                        worker_state,
                        *this);
                    workers.emplace_back(worker);
                }
            }

            template<typename TTask, typename Worker>
            WorkerPool<TTask, Worker>::~WorkerPool()
            {
            }

            template<typename TTask, typename Worker>
            void WorkerPool<TTask, Worker>::start()
            {
                for(auto& worker : workers)
                    worker->start();
            }

            template<typename TTask, typename Worker>
            void WorkerPool<TTask, Worker>::stop()
            {
                for(auto& worker : workers)
                    worker->stop();

                workers.clear();
            }

            template<typename TTask, typename Worker>
            int WorkerPool<TTask, Worker>::find_free_worker()
            {
                TRACE_EVENT("Scheduler", "find_worker");

                SPDLOG_TRACE("find worker...");

                unsigned start_idx = 0;
                if(TaskFreeCtx::current_worker_id)
                    start_idx = *TaskFreeCtx::current_worker_id - m_base_id;

                std::optional<unsigned> idx = this->probe_worker_by_state<unsigned>(
                    [this](unsigned idx) -> std::optional<unsigned>
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
