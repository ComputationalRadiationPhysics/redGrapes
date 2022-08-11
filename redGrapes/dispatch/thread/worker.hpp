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

#include <redGrapes/cv.hpp>

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
    std::atomic_bool m_stop;
    CondVar cv;

public:
    task::Queue queue;
    std::thread thread;

public:

    WorkerThread() :
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

                while( ! m_stop )
                {
                    SPDLOG_TRACE("Worker: take jobs");

                    while( 1)
                    {
                        Task * task = queue.pop();

                        if(task)
                        dispatch::thread::execute_task( *task , this->shared_from_this() );
                        else
                            break;
		    }

                    SPDLOG_TRACE("Worker: empty");
                    redGrapes::schedule();

                    SPDLOG_TRACE("Worker wait!");
                    cv.wait();
                }

                SPDLOG_TRACE("Worker Finished!");
            }
        )
    {
    }

    ~WorkerThread()
    {
        stop();
        thread.join();
    }

    bool wake()
    {
        SPDLOG_TRACE("Worker::wake()");
        return cv.notify();
    }

    void stop()
    {
        SPDLOG_TRACE("Worker::stop()");
        m_stop = true;
        wake();
    }
};

} // namespace thread
} // namespace dispatch
} // namespace redGrapes

