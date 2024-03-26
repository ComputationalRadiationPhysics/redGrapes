/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/dispatch/mpi/request_pool.hpp"
#include "redGrapes/dispatch/thread/WorkerThread.hpp"
#include "redGrapes/scheduler/scheduler.hpp"

#include <pthread.h>

#include <memory>

namespace redGrapes
{
    namespace scheduler
    {

        /*
         * Uses simple round-robin algorithm to distribute tasks to workers
         * and implements work-stealing
         */
        template<typename TTask, typename Worker>
        struct ThreadScheduler : public IScheduler<TTask>
        {
            WorkerId m_base_id;
            CondVar cv;
            std::shared_ptr<dispatch::thread::WorkerThread<TTask, Worker>> m_worker_thread;
            static constexpr unsigned n_workers = 1;

            ThreadScheduler()
            {
            }

            ThreadScheduler(std::shared_ptr<dispatch::thread::WorkerThread<TTask, Worker>> workerThread)
                : m_worker_thread(workerThread)
            {
            }

            void idle()
            {
                SPDLOG_TRACE("ThreadScheduler::idle()");

                /* the main thread shall not do any busy waiting
                 * and always sleep right away in order to
                 * not block any worker threads (those however should
                 * busy-wait to improve latency)
                 */
                cv.timeout = 0;
                cv.wait();
            }

            /* send the new task to a worker
             */
            void emplace_task(TTask& task)
            {
                // todo: properly store affinity information in task
                m_worker_thread->worker->dispatch_task(task);
            }

            /* send this already existing,
             * but only through follower-list so it is not assigned to a worker yet.
             * since this task is now ready, send find a worker for it
             */
            void activate_task(TTask& task)
            {
                //! worker id to use in case all workers are busy
                TRACE_EVENT("Scheduler", "activate_task");
                SPDLOG_TRACE("ThreadScheduler::activate_task({})", task.task_id);

                m_worker_thread->worker->ready_queue.push(&task);
                m_worker_thread->worker->wake();
            }

            /* Wakeup some worker or the main thread
             *
             * WakerId = 0 for main thread
             * WakerId = WorkerId + 1
             *
             * @return true if thread was indeed asleep
             */
            bool wake(WakerId id = 0)
            {
                if(id == 0)
                    return cv.notify();
                else if(id > 0 && id <= 1)
                    return m_worker_thread->worker->wake();
                else
                    return false;
            }

            /* wakeup all wakers (workers + main thread)
             */
            void wake_all()
            {
                cv.notify();
                m_worker_thread->worker->wake();
            }

            unsigned getNextWorkerID()
            {
                return m_base_id;
            }

            void init(WorkerId base_id)
            {
                m_base_id = base_id;
                // TODO check if it was already initalized
                if(!m_worker_thread)
                {
                    unsigned pu_id = base_id % TaskFreeCtx::n_pus;
                    // allocate worker with id `i` on arena `i`,
                    hwloc_obj_t obj = hwloc_get_obj_by_type(TaskFreeCtx::hwloc_ctx.topology, HWLOC_OBJ_PU, pu_id);
                    TaskFreeCtx::worker_alloc_pool->allocs.emplace_back(
                        memory::HwlocAlloc(TaskFreeCtx::hwloc_ctx, obj),
                        REDGRAPES_ALLOC_CHUNKSIZE);

                    m_worker_thread = memory::alloc_shared_bind<dispatch::thread::WorkerThread<TTask, Worker>>(
                        m_base_id,
                        TaskFreeCtx::worker_alloc_pool->get_alloc(m_base_id),
                        TaskFreeCtx::hwloc_ctx,
                        obj,
                        m_base_id);
                }
                // m_worker_pool->emplace_workers();
            }

            void startExecution()
            {
                m_worker_thread->start();
            }

            void stopExecution()
            {
                m_worker_thread->stop();
            }

            // if worker is  MPI worker
            std::shared_ptr<dispatch::mpi::RequestPool<TTask>> getRequestPool()
            {
                return m_worker_thread->worker->requestPool;
            }
        };


    } // namespace scheduler

} // namespace redGrapes
