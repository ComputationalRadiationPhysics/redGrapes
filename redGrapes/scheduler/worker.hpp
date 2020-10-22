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

struct IWorker
{
    virtual ~IWorker() {};
    virtual void work() = 0;
    virtual void notify() = 0;
    virtual void stop() = 0;
};

struct DefaultWorker : IWorker
{
private:
    std::mutex m;
    std::condition_variable cv;

    std::atomic_bool m_stop;
    std::atomic_flag wait = ATOMIC_FLAG_INIT;

    std::function< bool () > consume;

public:
    DefaultWorker( std::function< bool () > consume ) :
        m_stop( false ),
        consume( consume )
    {}
    
    void work()
    {
        while( ! m_stop )
        {
            while( consume() );

            std::unique_lock< std::mutex > l( m );
            spdlog::trace("Worker waiting..");
            cv.wait( l, [this]{ return !wait.test_and_set(); } );

            spdlog::trace("Worker continued.");
        }

        spdlog::trace("Worker Finished!");
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
        m_stop = true;
        notify();        
    }
};

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

