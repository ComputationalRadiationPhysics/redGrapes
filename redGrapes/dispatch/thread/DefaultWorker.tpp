/* Copyright 2020-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "redGrapes/TaskCtx.hpp"
#include "redGrapes/dispatch/thread/DefaultWorker.hpp"
#include "redGrapes/dispatch/thread/worker_pool.hpp"
#include "redGrapes/util/bitfield.hpp"
#include "redGrapes/util/trace.hpp"

#include <hwloc.h>

#include <atomic>

namespace redGrapes
{
    namespace dispatch
    {
        namespace thread
        {

            template<typename TTask>
            DefaultWorker<TTask>::~DefaultWorker()
            {
            }

            template<typename TTask>
            void DefaultWorker<TTask>::stop()
            {
                SPDLOG_TRACE("Worker::stop()");
                m_stop.store(true, std::memory_order_release);
                wake();
            }

            template<typename TTask>
            void DefaultWorker<TTask>::work_loop()
            {
                SPDLOG_TRACE("Worker {} start work_loop()", id);
                while(!m_stop.load(std::memory_order_consume))
                {
                    m_worker_state.set(id, dispatch::thread::WorkerState::AVAILABLE);
                    cv.wait();

                    while(TTask* task = this->gather_task())
                    {
                        m_worker_state.set(id, dispatch::thread::WorkerState::BUSY);
                        execute_task(*task);
                    }
                }
                SPDLOG_TRACE("Worker {} end work_loop()", id);
            }

            template<typename TTask>
            void DefaultWorker<TTask>::execute_task(TTask& task)
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

            template<typename TTask>
            TTask* DefaultWorker<TTask>::gather_task()
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

                /* set worker state to signal that we are requesting tasks
                 */
                m_worker_state.set(id, dispatch::thread::WorkerState::AVAILABLE);

#ifndef ENABLE_WORKSTEALING
#    define ENABLE_WORKSTEALING 1
#endif

#if ENABLE_WORKSTEALING

                /* STAGE 3:
                 *
                 * after all tasks from own queues are consumed, try to steal tasks
                 */
                SPDLOG_TRACE("Worker {}: try to steal tasks", id);
                task = m_worker_pool.steal_task(*this);

#endif

                return task;
            }

            template<typename TTask> // task with graphable
            bool DefaultWorker<TTask>::init_dependencies(TTask*& t, bool claimed)
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

        } // namespace thread
    } // namespace dispatch
} // namespace redGrapes
