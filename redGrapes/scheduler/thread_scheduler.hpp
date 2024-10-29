/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/dispatch/thread/WorkerThread.hpp"
#include "redGrapes/memory/allocator.hpp"
#include "redGrapes/scheduler/scheduler.hpp"

#include <memory>

namespace redGrapes
{
    namespace scheduler
    {

        template<typename Worker>
        struct ThreadScheduler : public IScheduler<typename Worker::task_type>
        {
            using TTask = Worker::task_type;

            WorkerId m_base_id;
            std::shared_ptr<dispatch::thread::WorkerThread<Worker>> m_worker_thread;
            static constexpr WorkerId n_workers = 1;

            ThreadScheduler()
            {
            }

            ThreadScheduler(std::shared_ptr<dispatch::thread::WorkerThread<Worker>> workerThread)
                : m_worker_thread(std::move(workerThread))
            {
            }

            /* send the new task to a worker
             */
            void emplace_task(TTask& task)
            {
                // todo: properly store affinity information in task
                m_worker_thread->worker.dispatch_task(task);
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

                m_worker_thread->worker.ready_queue.push(&task);
                m_worker_thread->worker.wake();
            }

            /* Wakeup some worker or the main thread
             *
             * WorkerId = WorkerId
             *
             * @return true if thread was indeed asleep
             */
            bool wake(WakerId id = 0)
            {
                // TODO remove this if else
                if(id == 0)
                    return m_worker_thread->worker.wake();
                else
                    return false;
            }

            /* wakeup all workers
             */
            void wake_all()
            {
                m_worker_thread->worker.wake();
            }

            WorkerId getNextWorkerID()
            {
                return m_base_id;
            }

            void init(WorkerId base_id) override
            {
                m_base_id = base_id;
                // TODO check if it was already initalized
                if(!m_worker_thread)
                {
                    WorkerId pu_id = base_id % TaskFreeCtx::n_pus;
                    // allocate worker with id `i` on arena `i`,
                    hwloc_obj_t obj = hwloc_get_obj_by_type(TaskFreeCtx::hwloc_ctx.topology, HWLOC_OBJ_PU, pu_id);
                    TaskFreeCtx::worker_alloc_pool.allocs.emplace_back(
                        memory::HwlocAlloc(TaskFreeCtx::hwloc_ctx, obj),
                        REDGRAPES_ALLOC_CHUNKSIZE);

                    m_worker_thread
                        = memory::alloc_shared_bind<dispatch::thread::WorkerThread<Worker>>(m_base_id, obj, m_base_id);
                }
            }

            void startExecution()
            {
                m_worker_thread->start();
            }

            void stopExecution()
            {
                m_worker_thread->stop();
            }
        };


    } // namespace scheduler

} // namespace redGrapes
