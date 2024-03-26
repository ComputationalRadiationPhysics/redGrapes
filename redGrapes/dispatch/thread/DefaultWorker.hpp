/* Copyright 2020-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/scheduler/scheduler.hpp"
#include "redGrapes/sync/cv.hpp"
#include "redGrapes/task/queue.hpp"
#include "redGrapes/util/bitfield.hpp"

#include <hwloc.h>
#include <moodycamel/concurrentqueue.h>

#include <atomic>

namespace redGrapes
{

    namespace dispatch
    {
        namespace thread
        {

            template<typename TTask, typename Worker>
            struct WorkerPool;

            /*!
             * Creates a thread which repeatedly calls consume()
             * until stop() is invoked or the object destroyed.
             *
             * Sleeps when no jobs are available.
             */
            template<typename TTask>
            struct DefaultWorker
            {
                // private:
                WorkerId id;
                AtomicBitfield& m_worker_state;
                WorkerPool<TTask, DefaultWorker>& m_worker_pool;

                /*! if true, the thread shall stop
                 * instead of waiting when it is out of jobs
                 */
                std::atomic_bool m_stop{false};
                std::atomic<unsigned> task_count{0};

                //! condition variable for waiting if queue is empty
                CondVar cv;

                static constexpr size_t queue_capacity = 128;

            public:
                task::Queue<TTask> emplacement_queue{queue_capacity};
                task::Queue<TTask> ready_queue{queue_capacity};

                DefaultWorker(
                    WorkerId worker_id,
                    AtomicBitfield& worker_state,
                    WorkerPool<TTask, DefaultWorker>& worker_pool)
                    : id(worker_id)
                    , m_worker_state(worker_state)
                    , m_worker_pool(worker_pool)
                {
                }

                ~DefaultWorker();

                inline scheduler::WakerId get_waker_id()
                {
                    return id + 1;
                }

                inline bool wake()
                {
                    return cv.notify();
                }

                void stop();

                /* adds a new task to the emplacement queue
                 * and wakes up thread to kickstart execution
                 */
                inline void dispatch_task(TTask& task)
                {
                    emplacement_queue.push(&task);
                    wake();
                }

                inline void execute_task(TTask& task);

                // private:

                /* repeatedly try to find and execute tasks
                 * until stop-flag is triggered by stop()
                 */
                void work_loop();

                /* find a task that shall be executed next
                 */
                TTask* gather_task();

                /*! take a task from the emplacement queue and initialize it,
                 * @param t is set to the task if the new task is ready,
                 * @param t is set to nullptr if the new task is blocked.
                 * @param claimed if set, the new task will not be actiated,
                 *        if it is false, activate_task will be called by notify_event
                 *
                 * @return false if queue is empty
                 */
                bool init_dependencies(TTask*& t, bool claimed = true);
            };

        } // namespace thread
    } // namespace dispatch
} // namespace redGrapes
