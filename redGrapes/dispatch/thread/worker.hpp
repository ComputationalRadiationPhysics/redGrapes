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
#include <redGrapes/util/trace.hpp>

namespace redGrapes
{
namespace dispatch
{
namespace thread
{

void execute_task( Task & task_id );

extern thread_local scheduler::WakerID current_waker_id;

/*!
 * Creates a thread which repeatedly calls consume()
 * until stop() is invoked or the object destroyed.
 *
 * Sleeps when no jobs are available.
 */
struct WorkerThread : std::enable_shared_from_this<WorkerThread>
{
private:

    /*!
     * if true, the thread shall start
     * executing the jobs in its queue
     * and request rescheduling if empty
     */
    std::atomic_bool m_start;
    
    /*! if true, the thread shall stop
     * instead of waiting when it is out of jobs
     */
    std::atomic_bool m_stop;

    //! condition variable for waiting if queue is empty
    CondVar cv;

public:
    std::shared_ptr< task::Queue > emplacement_queue;
    std::shared_ptr< task::Queue > ready_queue;
    std::thread thread;

    scheduler::WakerID id;

public:
    WorkerThread( scheduler::WakerID id ) :
        m_start( false ),
        m_stop( false ),
        id( id ),
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
                
                current_waker_id = this->id;

                emplacement_queue = std::make_shared< task::Queue >( 32 );
                ready_queue = std::make_shared< task::Queue >( 32 );

                while( ! m_start.load(std::memory_order_consume) )
                    cv.wait();

                while( ! m_stop.load(std::memory_order_consume) )
                {
                    SPDLOG_TRACE("Worker: work on queue");

                    Task * task;
                    
                    while( task = ready_queue->pop() )
                        dispatch::thread::execute_task( *task );

                    if( task = redGrapes::schedule( *this ) )
                        dispatch::thread::execute_task( *task );
                    else if( !m_stop.load(std::memory_order_consume) )
                    {
                        SPDLOG_TRACE("worker sleep");
                        //TRACE_EVENT("Worker", "sleep");
                        cv.wait();
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

    inline unsigned get_worker_id()
    {
        return id - 1;
    }
    
    inline scheduler::WakerID get_waker_id()
    {
        return id;
    }

    inline bool wake()
    {
        return cv.notify();
    }

    void start()
    {
        m_start.store(true, std::memory_order_release);
        wake();
    }

    void stop()
    {
        SPDLOG_TRACE("Worker::stop()");
        m_stop.store(true, std::memory_order_release);
        wake();
        thread.join();
    }

    void emplace_task( Task * task )
    {
        emplacement_queue->push( task );
        wake();
    }

    /*! take a task from the emplacement queue and initialize it,
     * @param t is set to the task if the new task is ready,
     * @param t is set to nullptr if the new task is blocked.
     * @param claimed if set, the new task will not be actiated,
     *        if it is false, activate_task will be called by notify_event
     *
     * @return false if queue is empty
     */
    bool init_dependencies( Task* & t, bool claimed = true )
    {
        if(Task * task = emplacement_queue->pop())
        {
            SPDLOG_DEBUG("init task {}", task->task_id);

            task->pre_event.up();
            task->init_graph();

            if( task->get_pre_event().notify( claimed ) )
                t = task;
            else
            {
                //s           spdlog::info("task already taken");
                t = nullptr;
            }

            return true;
        }
        else
            return false;
    }
};

} // namespace thread
} // namespace dispatch
} // namespace redGrapes

