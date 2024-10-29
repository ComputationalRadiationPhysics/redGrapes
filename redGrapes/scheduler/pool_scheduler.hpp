/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/dispatch/thread/worker_pool.hpp"
#include "redGrapes/scheduler/scheduler.hpp"

namespace redGrapes
{
    namespace scheduler
    {

        /*
         * Uses simple round-robin algorithm to distribute tasks to workers
         * and implements work-stealing
         */
        template<typename Worker>
        struct PoolScheduler : public IScheduler<typename Worker::task_type>
        {
            using TTask = Worker::task_type;
            WorkerId m_base_id;
            WorkerId n_workers;
            dispatch::thread::WorkerPool<Worker> m_worker_pool;

            PoolScheduler(WorkerId num_workers);
            PoolScheduler(dispatch::thread::WorkerPool<Worker> workerPool);

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
             * WakerId = WorkerId
             *
             * @return true if thread was indeed asleep
             */
            bool wake(WakerId id);

            /* wakeup all workers
             */
            void wake_all();

            WorkerId getNextWorkerID();

            void init(WorkerId base_id) override;

            void startExecution();

            void stopExecution();
        };

    } // namespace scheduler

} // namespace redGrapes

#include "redGrapes/scheduler/pool_scheduler.tpp"
