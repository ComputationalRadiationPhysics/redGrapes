/* Copyright 2022-2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <redGrapes/util/bitfield.hpp>
#include <redGrapes/dispatch/thread/worker.hpp>

namespace redGrapes
{
namespace dispatch
{
namespace thread
{

struct WorkerPool
{
    WorkerPool( size_t n_workers = 1 );

    /* get the number of workers in this pool
     */
    inline size_t size()
    {
        return workers.size();
    }

    /* signals all workers to start executing tasks
     */
    void start();

    /* signals all workers that no new tasks will be added
     */
    void stop();

    inline WorkerThread & get_worker( WorkerId worker_id )
    {
        return *workers[ worker_id ];
    }

    inline WorkerState get_worker_state( WorkerId worker_id )
    {
        return worker_state.get(worker_id) ? WorkerState::AVAILABLE : WorkerState::BUSY;
    }

    /* return true on success
     */
    inline bool set_worker_state( WorkerId worker_id, WorkerState state )
    {
        return worker_state.set( worker_id, state ) != state;
    }

    template <typename T, typename F>
    inline std::optional< T >
    probe_worker_by_state(
        F && f,
        bool expected_worker_state,
        unsigned start_worker_idx,
        bool exclude_start = true)
    {
        return worker_state.template probe_by_value<T, F>( std::move(f), expected_worker_state, start_worker_idx );
    }

    /*!
     * tries to find an available worker, but potentially
     * returns a busy worker if no free worker is available
     *     
     * @return worker_id
     */
    int find_free_worker();
    
private:
    std::vector< std::shared_ptr<dispatch::thread::WorkerThread> > workers;
    AtomicBitfield worker_state;
};

} // namespace thread
} // namespace dispatch
} // namespace redGrapes

