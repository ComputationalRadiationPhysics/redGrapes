/* Copyright 2022-2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <redGrapes/dispatch/thread/worker.hpp>
#include <redGrapes/dispatch/thread/worker_pool.hpp>
#include <redGrapes/memory/allocator.hpp>
#include <redGrapes/memory/chunked_bump_alloc.hpp>
#include <redGrapes/memory/hwloc_alloc.hpp>
#include <redGrapes/redGrapes.hpp>

// #include <redGrapes_config.hpp>

namespace redGrapes
{
    namespace dispatch
    {
        namespace thread
        {

            WorkerPool::WorkerPool(HwlocContext& hwloc_ctx, size_t n_workers)
                : hwloc_ctx(hwloc_ctx)
                , worker_state(n_workers)
            {
                Context::current_waker_id = 0;
            }

            void WorkerPool::emplace_workers(size_t n_workers)
            {
                unsigned n_pus = hwloc_get_nbobjs_by_type(hwloc_ctx.topology, HWLOC_OBJ_PU);
                if(n_workers > n_pus)
                    spdlog::warn("{} worker-threads requested, but only {} PUs available!", n_workers, n_pus);

                allocs.reserve(n_workers);
                workers.reserve(n_workers);

                SPDLOG_INFO("populate WorkerPool with {} workers", n_workers);
                for(size_t worker_id = 0; worker_id < n_workers; ++worker_id)
                {
                    unsigned pu_id = worker_id % n_pus;
                    // allocate worker with id `i` on arena `i`,
                    hwloc_obj_t obj = hwloc_get_obj_by_type(hwloc_ctx.topology, HWLOC_OBJ_PU, pu_id);
                    allocs.emplace_back(memory::HwlocAlloc(hwloc_ctx, obj), REDGRAPES_ALLOC_CHUNKSIZE);

                    SingletonContext::get().current_arena = pu_id;
                    auto worker
                        = memory::alloc_shared_bind<WorkerThread>(pu_id, get_alloc(pu_id), hwloc_ctx, obj, worker_id);
                    //        auto worker = std::make_shared< WorkerThread >( get_alloc(i), hwloc_ctx, obj, i );
                    workers.emplace_back(worker);
                }
            }

            WorkerPool::~WorkerPool()
            {
            }

            void WorkerPool::start()
            {
                for(auto& worker : workers)
                    worker->start();
            }

            void WorkerPool::stop()
            {
                for(auto& worker : workers)
                    worker->stop();

                workers.clear();
            }

            int WorkerPool::find_free_worker()
            {
                TRACE_EVENT("Scheduler", "find_worker");

                SPDLOG_TRACE("find worker...");

                unsigned start_idx = 0;
                if(auto w = SingletonContext::get().current_worker)
                    start_idx = w->get_worker_id();

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
