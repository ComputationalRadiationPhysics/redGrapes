/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/dispatch/thread/DefaultWorker.hpp"
#include "redGrapes/scheduler/pool_scheduler.hpp"
#include "redGrapes/util/trace.hpp"

#include <spdlog/spdlog.h>

namespace redGrapes
{
    namespace scheduler
    {
        template<typename TTask, typename Worker>
        PoolScheduler<TTask, Worker>::PoolScheduler(unsigned num_workers)
            : n_workers(num_workers)
            , m_worker_pool(
                  std::make_shared<dispatch::thread::WorkerPool<TTask, Worker>>(TaskFreeCtx::hwloc_ctx, num_workers))
        {
        }

        template<typename TTask, typename Worker>
        PoolScheduler<TTask, Worker>::PoolScheduler(
            std::shared_ptr<dispatch::thread::WorkerPool<TTask, Worker>> workerPool)
            : m_worker_pool(workerPool)
        {
        }

        template<typename TTask, typename Worker>
        void PoolScheduler<TTask, Worker>::idle()
        {
            SPDLOG_TRACE("PoolScheduler::idle()");

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
        template<typename TTask, typename Worker>
        void PoolScheduler<TTask, Worker>::emplace_task(TTask& task)
        {
            // TODO: properly store affinity information in task
            WorkerId worker_id = task.worker_id - m_base_id;

            m_worker_pool->get_worker_thread(worker_id).worker->dispatch_task(task);

            /* hack as of 2023/11/17
             *
             * Additionally to the worker who got the new task above,
             * we will now notify another, available (idling) worker,
             * in trying to avoid stale tasks in cases where new tasks
             * are assigned to an already busy worker.
             */
#ifndef REDGRAPES_EMPLACE_NOTIFY_NEXT
#    define REDGRAPES_EMPLACE_NOTIFY_NEXT 0
#endif

#if REDGRAPES_EMPLACE_NOTIFY_NEXT
            auto id = m_worker_pool->probe_worker_by_state<unsigned>(
                [&m_worker_pool](unsigned idx)
                {
                    m_worker_pool->get_worker_thread(idx).worker->wake();
                    return idx;
                },
                dispatch::thread::WorkerState::AVAILABLE,
                worker_id,
                true);
#endif
        }

        /* send this already existing task to a worker,
         * but only through follower-list so it is not assigned to a worker yet.
         * since this task is now ready, send find a worker for it
         */
        template<typename TTask, typename Worker>
        void PoolScheduler<TTask, Worker>::activate_task(TTask& task)
        {
            //! worker id to use in case all workers are busy
            // TODO analyse and optimize
            static thread_local std::atomic<unsigned int> next_worker(
                TaskFreeCtx::current_worker_id ? *TaskFreeCtx::current_worker_id + 1 - m_base_id : 0);
            TRACE_EVENT("Scheduler", "activate_task");
            SPDLOG_TRACE("PoolScheduler::activate_task({})", task.task_id);

            int worker_id = m_worker_pool->find_free_worker();
            if(worker_id < 0)
            {
                worker_id = next_worker.fetch_add(1) % n_workers;
                if(worker_id == *TaskFreeCtx::current_worker_id)
                    worker_id = next_worker.fetch_add(1) % n_workers;
            }

            m_worker_pool->get_worker_thread(worker_id).worker->ready_queue.push(&task);
            m_worker_pool->set_worker_state(worker_id, dispatch::thread::WorkerState::BUSY);
            m_worker_pool->get_worker_thread(worker_id).worker->wake();
        }

        /* Wakeup some worker or the main thread
         *
         * WakerId = 0 for main thread
         * WakerId = WorkerId + 1
         *
         * @return true if thread was indeed asleep
         */
        template<typename TTask, typename Worker>
        bool PoolScheduler<TTask, Worker>::wake(WakerId id)
        {
            if(id == 0)
                return cv.notify();
            // TODO analyse and optimize
            else if(id > 0 && id - m_base_id <= n_workers)
                return m_worker_pool->get_worker_thread(id - m_base_id - 1).worker->wake();
            else
                return false;
        }

        /* wakeup all wakers (workers + main thread)
         */
        template<typename TTask, typename Worker>
        void PoolScheduler<TTask, Worker>::wake_all()
        {
            wake(0);
            for(uint16_t i = m_base_id; i < m_base_id + n_workers; ++i)
                wake(i);
        }

        template<typename TTask, typename Worker>
        unsigned PoolScheduler<TTask, Worker>::getNextWorkerID()
        {
            // TODO make atomic
            auto id = local_next_worker_id + m_base_id;
            local_next_worker_id = (local_next_worker_id + 1) % n_workers;
            return id;
        }

        template<typename TTask, typename Worker>
        void PoolScheduler<TTask, Worker>::init(WorkerId base_id)
        {
            // TODO check if it was already initalized
            m_base_id = base_id;
            m_worker_pool->emplace_workers(m_base_id);
        }

        template<typename TTask, typename Worker>
        void PoolScheduler<TTask, Worker>::startExecution()
        {
            // TODO check if it was already started
            m_worker_pool->start();
        }

        template<typename TTask, typename Worker>
        void PoolScheduler<TTask, Worker>::stopExecution()
        {
            // TODO check if it was already stopped
            m_worker_pool->stop();
        }


    } // namespace scheduler
} // namespace redGrapes
