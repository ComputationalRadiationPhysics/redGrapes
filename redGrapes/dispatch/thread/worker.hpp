/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <thread>
#include <atomic>
#include <functional>
#include <memory>
#include <moodycamel/concurrentqueue.h>

#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/context.hpp>
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/util/cv.hpp>

namespace redGrapes
{
namespace dispatch
{
namespace thread
{

void execute_task( Task & task, std::weak_ptr<scheduler::IWaker> waker = std::weak_ptr<scheduler::IWaker>() );

/*!
 * Creates a thread which repeatedly calls consume()
 * until stop() is invoked or the object destroyed.
 *
 * Sleeps when no jobs are available.
 */
struct WorkerThread : virtual scheduler::IWaker, std::enable_shared_from_this<WorkerThread>
{
private:

    /*! if true, the thread shall stop
     * instead of waiting when it is out of jobs
     */
    std::atomic_bool m_start;
    std::atomic_bool m_stop;
    CondVar cv;

public:
    task::Queue queue;
    std::thread thread;

    std::atomic_bool has_work;

public:

    WorkerThread() :
        m_start( false ),
        m_stop( false ),
        thread(
            [this]
            {
                /* since we are in a worker, there should always
                 * be a task running (we always have a parent task
                 * and therefore yield() guarantees to do
                 * a context-switch instead of idling
                 */
                redGrapes::idle =
                    [this]
                    {
                        throw std::runtime_error("idle in worker thread!");
                    };

                while( ! m_start )
                    cv.wait();

                while( ! m_stop )
                {
                    SPDLOG_DEBUG("Worker: work on queue");
                  
                    while( Task * task = queue.pop() )
                        dispatch::thread::execute_task( *task , this->shared_from_this() );

                    has_work.exchange(false);
                    redGrapes::schedule( *this );

                    if( !m_stop && !has_work )
                    {
                        SPDLOG_DEBUG("Worker: queue empty -> wait");
                        cv.wait();
                        SPDLOG_DEBUG("Wake!");
                    }
                }

                SPDLOG_TRACE("Worker Finished!");
            }
        )
    {
    }

    ~WorkerThread()
    {
    }

    bool wake()
    {
        bool awake = cv.notify();
        SPDLOG_TRACE("Worker::wake() ... awake={}", awake);
        return awake;
    }

    void start()
    {
        m_start = true;
        wake();
    }

    void stop()
    {
        SPDLOG_TRACE("Worker::stop()");
        m_stop = true;
        wake();
        thread.join();
    }
};

} // namespace thread
} // namespace dispatch
} // namespace redGrapes

