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

#include <redGrapes/task/task_base.hpp>

namespace redGrapes
{
namespace dispatch
{
namespace thread
{

template < typename Task, typename TaskVertexPtr >
void execute_task( IManager & rg, TaskVertexPtr task_vertex )
{
    auto & task = task_vertex->template get_task<Task>();
    assert( task.is_ready() );

    SPDLOG_TRACE("thread dispatch: execute task {}", task.task_id);

    rg.notify_event( scheduler::EventPtr{ scheduler::T_EVT_PRE, task_vertex } );

    scope_level = task.scope_level;
    rg.current_task() = task_vertex;

    if( auto event = task() )
    {
        //task.sg_pause( *event );

        task.pre_event.up();
        rg.notify_event( scheduler::EventPtr{ scheduler::T_EVT_PRE, task_vertex } );
    }
    else
        rg.notify_event( scheduler::EventPtr{ scheduler::T_EVT_POST, task_vertex } );

    rg.current_task() = std::nullopt;
}

/*!
 * Creates a thread which repeatedly calls consume()
 * until stop() is invoked or the object destroyed.
 *
 * Sleeps when no jobs are available.
 */
template < typename Task >
struct WorkerThread
{
private:
    IManager & mgr;
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
    WorkerThread( IManager & mgr, std::shared_ptr< scheduler::IScheduler > scheduler ) :
        m_stop( false ),
        mgr( mgr ),
        scheduler( scheduler ),
        thread(
            [this]
            {
                /* since we are in a worker, there should always
                 * be a task running (we always have a parent task
                 * and therefore yield() guarantees to do
                 * a context-switch instead of idling
                 */
                redGrapes::dispatch::thread::idle =
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
                        dispatch::thread::execute_task<Task>( this->mgr, *task );
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

