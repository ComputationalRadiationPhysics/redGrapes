/* Copyright 2022-2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <redGrapes/dispatch/thread/worker.hpp>
#include <redGrapes/dispatch/thread/worker_pool.hpp>
#include <redGrapes/util/chunked_bump_alloc.hpp>

namespace redGrapes
{
namespace dispatch
{
namespace thread
{

WorkerPool::WorkerPool( size_t n_workers )
    : worker_state( n_workers )
{
    workers.reserve( n_workers );

    unsigned n_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
    if( n_workers > n_pus )
        spdlog::warn("{} worker-threads requested, but only {} PUs available!", n_workers, n_pus);

    SPDLOG_INFO("create WorkerPool with {} workers", n_workers);
    for( size_t i = 0; i < n_workers; ++i )
    {
        // allocate worker with id `i` on arena `i`,
        hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
        memory::HwlocAlloc< dispatch::thread::WorkerThread > hwloc_alloc( obj );
        auto worker = std::allocate_shared< dispatch::thread::WorkerThread >( hwloc_alloc, obj, i );
        workers.emplace_back( worker );
    }

    redGrapes::dispatch::thread::current_waker_id = 0;
}

WorkerPool::~WorkerPool()
{
   SPDLOG_TRACE("~WorkerPool()");
}

void WorkerPool::start()
{
    for( auto & worker : workers )
        worker->start();
}

void WorkerPool::stop()
{
    for( auto & worker : workers )
        worker->stop();
}

int WorkerPool::find_free_worker()
{
    TRACE_EVENT("Scheduler", "find_worker");

    SPDLOG_TRACE("find worker...");

    unsigned start_idx = 0;
    if(auto w = dispatch::thread::current_worker)
        start_idx = w->get_worker_id();

    std::optional<unsigned> idx =
        this->probe_worker_by_state<unsigned>(
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

    if( idx )
        return *idx;
    else
        // no free worker found,
        return -1;
}

} // namespace thread
} // namespace dispatch
} // namespace redGrapes

