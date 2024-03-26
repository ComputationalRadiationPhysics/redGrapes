/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once
#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/dispatch/mpi/request_pool.hpp"
#include "redGrapes/sync/cv.hpp"
#include "redGrapes/task/queue.hpp"

#include <memory>

namespace redGrapes
{
    namespace dispatch
    {
        namespace mpi
        {

            template<typename TTask>
            struct MPIWorker
            {
                std::shared_ptr<RequestPool<TTask>> requestPool;
                WorkerId id;

                /*! if true, the thread shall stop
                 * instead of waiting when it is out of jobs
                 */
                std::atomic_bool m_stop{false};
                std::atomic<unsigned> task_count{0};

                //! condition variable for waiting if queue is empty
                CondVar cv;

                static constexpr size_t queue_capacity = 128;
                task::Queue<TTask> emplacement_queue{queue_capacity};
                task::Queue<TTask> ready_queue{queue_capacity};

                MPIWorker(WorkerId worker_id) : id(worker_id)
                {
                    requestPool = std::make_shared<RequestPool<TTask>>();
                }

                ~MPIWorker()
                {
                }

                inline scheduler::WakerId get_waker_id()
                {
                    return id + 1;
                }

                inline bool wake()
                {
                    return cv.notify();
                }

                void stop()
                {
                    SPDLOG_TRACE("Worker::stop()");
                    m_stop.store(true, std::memory_order_release);
                    wake();
                }

                /* adds a new task to the emplacement queue
                 * and wakes up thread to kickstart execution
                 */
                inline void dispatch_task(TTask& task)
                {
                    emplacement_queue.push(&task);
                    wake();
                }

                inline void execute_task(TTask& task)
                {
                    TRACE_EVENT("Worker", "dispatch task");

                    SPDLOG_DEBUG("thread dispatch: execute task {}", task.task_id);
                    assert(task.is_ready());

                    task.get_pre_event().notify();
                    TaskCtx<TTask>::current_task = &task;

                    auto event = task();

                    if(event)
                    {
                        event->get_event().waker_id = get_waker_id();
                        task.sg_pause(*event);

                        task.pre_event.up();
                        task.get_pre_event().notify();
                    }
                    else
                        task.get_post_event().notify();

                    TaskCtx<TTask>::current_task = nullptr;
                }

                /* find a task that shall be executed next
                 */
                TTask* gather_task()
                {
                    {
                        TRACE_EVENT("Worker", "gather_task()");
                        TTask* task = nullptr;

                        /* STAGE 1:
                         *
                         * first, execute all tasks in the ready queue
                         */
                        SPDLOG_TRACE("Worker {}: consume ready queue", id);
                        if((task = ready_queue.pop()))
                            return task;

                        /* STAGE 2:
                         *
                         * after the ready queue is fully consumed,
                         * try initializing new tasks until one
                         * of them is found to be ready
                         */
                        SPDLOG_TRACE("Worker {}: try init new tasks", id);
                        while(this->init_dependencies(task, true))
                            if(task)
                                return task;

                        return task;
                    }
                }

                /*! take a task from the emplacement queue and initialize it,
                 * @param t is set to the task if the new task is ready,
                 * @param t is set to nullptr if the new task is blocked.
                 * @param claimed if set, the new task will not be actiated,
                 *        if it is false, activate_task will be called by notify_event
                 *
                 * @return false if queue is empty
                 */
                bool init_dependencies(TTask*& t, bool claimed = true)
                {
                    {
                        TRACE_EVENT("Worker", "init_dependencies()");
                        if(TTask* task = emplacement_queue.pop())
                        {
                            SPDLOG_DEBUG("init task {}", task->task_id);

                            task->pre_event.up();
                            task->init_graph();

                            if(task->get_pre_event().notify(claimed))
                                t = task;
                            else
                            {
                                t = nullptr;
                            }

                            return true;
                        }
                        else
                            return false;
                    }
                }

                /* repeatedly try to find and execute tasks
                 * until stop-flag is triggered by stop()
                 */
                void work_loop()
                {
                    SPDLOG_TRACE("Worker {} start work_loop()", this->id);
                    while(!this->m_stop.load(std::memory_order_consume))
                    {
                        // this->cv.wait(); // TODO fix this by fixing event_ptr notify to wake

                        while(TTask* task = this->gather_task())
                        {
                            execute_task(*task);
                            requestPool->poll(); // TODO fix where to poll
                        }
                        requestPool->poll();
                    }
                    SPDLOG_TRACE("Worker {} end work_loop()", this->id);
                }
            };

        } // namespace mpi
    } // namespace dispatch
} // namespace redGrapes
