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

#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/context.hpp>

namespace redGrapes
{
namespace dispatch
{
namespace thread
{

void execute_task( std::shared_ptr< Task > task );

/*!
 * Creates a thread which repeatedly calls consume()
 * until stop() is invoked or the object destroyed.
 *
 * Sleeps when no jobs are available.
 */
struct WorkerThread
{
private:
    std::shared_ptr< scheduler::IScheduler > scheduler;

    /*! if true, the thread shall stop
     * instead of waiting when consume() is out of jobs
     */
    std::atomic_bool m_stop;

    //! is set when the worker thread is currently waiting
    std::atomic_flag wait = ATOMIC_FLAG_INIT;

    std::mutex m;
    std::condition_variable cv;

public:
    std::thread thread;

public:
    /*!
     * @param consume function that executes a task if possible and returns
     *                if any work is left
     */
    WorkerThread( std::shared_ptr< scheduler::IScheduler > scheduler ) :
        m_stop( false ),
        scheduler( scheduler ),
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
                    SPDLOG_TRACE("Worker Thread: sleep");
                    std::unique_lock< std::mutex > l( m );
                    cv.wait( l, [this]{ return !wait.test_and_set(); } );
                    l.unlock();
                    while( auto task = this->scheduler->get_job() )
                        dispatch::thread::execute_task( task );
                }

                SPDLOG_TRACE("Worker Finished!");
            }
        )
    {}

    ~WorkerThread()
    {
        stop();
        thread.join();
    }
    
    void notify()
    {
        std::unique_lock< std::mutex > l( m );
        wait.clear();
        l.unlock();

        cv.notify_one();
    }

    void stop()
    {
        SPDLOG_TRACE("Worker::stop()");
        m_stop = true;
        notify();
    }
};

} // namespace thread
} // namespace dispatch
} // namespace redGrapes

