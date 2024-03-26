/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/dispatch/thread/worker_pool.hpp"
#include "redGrapes/scheduler/scheduler.hpp"
#include "redGrapes/sync/cv.hpp"

#include <redGrapes/TaskFreeCtx.hpp>

#include <pthread.h>

namespace redGrapes
{
    namespace scheduler
    {

        /*
         * Uses simple round-robin algorithm to distribute tasks to workers
         * and implements work-stealing
         */
        template<typename TTask, typename Worker>
        struct PoolScheduler : public IScheduler<TTask>
        {
            WorkerId m_base_id;
            CondVar cv;
            WorkerId local_next_worker_id = 0;
            unsigned n_workers;
            std::shared_ptr<dispatch::thread::WorkerPool<TTask, Worker>> m_worker_pool;

            PoolScheduler(unsigned num_workers);
            PoolScheduler(std::shared_ptr<dispatch::thread::WorkerPool<TTask, Worker>> workerPool);

            void idle();

            /* send the new task to a worker
             */
            void emplace_task(TTask& task);

            /* send this already existing,
             * but only through follower-list so it is not assigned to a worker yet.
             * since this task is now ready, send find a worker for it
             */
            void activate_task(TTask& task);

            /* Wakeup some worker or the main thread
             *
             * WakerId = 0 for main thread
             * WakerId = WorkerId + 1
             *
             * @return true if thread was indeed asleep
             */
            bool wake(WakerId id = 0);

            /* wakeup all wakers (workers + main thread)
             */
            void wake_all();

            unsigned getNextWorkerID();

            void init(WorkerId base_id);

            void startExecution();

            void stopExecution();
        };

    } // namespace scheduler

} // namespace redGrapes
