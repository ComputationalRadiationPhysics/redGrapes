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

namespace redGrapes
{
namespace scheduler
{

/*! Worker Interface
 */
struct IWorker
{
    virtual ~IWorker() {};

    /*! Start a loop consuming work until stop() is called.
     */
    virtual void work() = 0;

    /*! Notify worker about potentially new work.
     * Wakes up worker thread if it was suspended previously.
     */
    virtual void notify() = 0;

    /*! Causes work() to return.
     */
    virtual void stop() = 0;
};

/*! Worker that sleeps when no work is available.
 */
struct DefaultWorker : IWorker
{
private:
    std::mutex m;
    std::condition_variable cv;

    std::atomic_bool m_stop;
    std::atomic_flag wait = ATOMIC_FLAG_INIT;

    std::function< bool () > consume;

public:
    /*!
     * @param consume function that executes a task if possible and returns
     *                if any work is left
     */
    DefaultWorker( std::function< bool () > consume ) :
        m_stop( false ),
        consume( consume )
    {}
    
    void work()
    {
        while( ! m_stop )
        {
            {
                std::unique_lock< std::mutex > l( m );
                cv.wait( l, [this]{ return !wait.test_and_set(); } );
            }

            while( consume() );
        }

        SPDLOG_TRACE("Worker Finished!");
    }

    void notify()
    {
        {
            std::unique_lock< std::mutex > l( m );
            wait.clear();
        }
        cv.notify_one();
    }

    void stop()
    {
        SPDLOG_TRACE("Worker::stop()");
        m_stop = true;
        notify();
    }
};

/*! Creates a thread which runs the work() function of Worker.
 *
 * @tparam Worker must satisfy the IWorker concept/interface
 */
template < typename Worker = DefaultWorker >
struct WorkerThread
{
    Worker worker;
    std::thread thread;

    WorkerThread( std::function< bool () > consume ) :
        worker( consume ),
        thread(
            [this]
            {
                /* since we are in a worker, there should always
                 * be a task running (we always have a parent task
                 * and therefore yield() guarantees to do
                 * a context-switch instead of idling
                 */
                redGrapes::thread::idle =
                    [this]
                    {
                        throw std::runtime_error("idle in worker thread!");
                    };

                this->worker.work();
            }
        )
    {}

    ~WorkerThread()
    {
        worker.stop();
        thread.join();
    }
};

} // namespace scheduler

} // namespace redGrapes

