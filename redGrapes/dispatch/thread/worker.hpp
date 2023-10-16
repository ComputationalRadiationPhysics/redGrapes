/* Copyright 2020-2023 Michael Sippel
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

#include <redGrapes/task/queue.hpp>
#include <redGrapes/util/cv.hpp>
#include <redGrapes/util/trace.hpp>

namespace redGrapes
{

extern hwloc_topology_t topology;

namespace dispatch
{
namespace thread
{

using WorkerId = unsigned;
enum WorkerState {
    BUSY = 0,
    AVAILABLE = 1
};


struct WorkerThread;

void execute_task( Task & task_id );

extern thread_local scheduler::WakerId current_waker_id;
extern thread_local std::shared_ptr< WorkerThread > current_worker;

/*!
 * Creates a thread which repeatedly calls consume()
 * until stop() is invoked or the object destroyed.
 *
 * Sleeps when no jobs are available.
 */
struct Worker
    : redGrapes::scheduler::IScheduler
{
    //private:
    WorkerId id;

    /*!
     * if true, the thread shall start
     * executing the jobs in its queue
     * and request rescheduling if empty
     */
    std::atomic_bool m_start{ false };
    
    /*! if true, the thread shall stop
     * instead of waiting when it is out of jobs
     */
    std::atomic_bool m_stop{ false };


    std::atomic<unsigned> task_count{ 0 };

    //! condition variable for waiting if queue is empty
    CondVar cv;

    static constexpr size_t queue_capacity = 32;

public:
    task::Queue emplacement_queue{ queue_capacity };
    task::Queue ready_queue{ queue_capacity };

    Worker( WorkerId id );
    virtual ~Worker();

    inline WorkerId get_worker_id() { return id; }
    inline scheduler::WakerId get_waker_id() { return id + 1; }
    inline bool wake() { return cv.notify(); }

    virtual void start();
    virtual void stop();

    /* adds a new task to the emplacement queue
     * and wakes up thread to kickstart execution
     */
    void emplace_task( Task & task )
    {
        emplacement_queue.push( &task );
        wake();
    }

    void activate_task( Task & task )
    {
        ready_queue.push( &task );
        wake();
    }

    //private:
    
    /* repeatedly try to find and execute tasks
     * until stop-flag is triggered by stop()
     */
    void work_loop();

    /* find a task that shall be executed next
     */
    Task * gather_task();
    
    /*! take a task from the emplacement queue and initialize it,
     * @param t is set to the task if the new task is ready,
     * @param t is set to nullptr if the new task is blocked.
     * @param claimed if set, the new task will not be actiated,
     *        if it is false, activate_task will be called by notify_event
     *
     * @return false if queue is empty
     */
    bool init_dependencies( Task* & t, bool claimed = true );
};

struct WorkerThread
    : Worker
    , std::enable_shared_from_this<WorkerThread>
{
    std::thread thread;

    WorkerThread( WorkerId worker_id );
    ~WorkerThread();

    void stop();

    /* function the thread will execute
     */
    void run();
    
    void cpubind();
    void membind();
};

} // namespace thread
} // namespace dispatch
} // namespace redGrapes

