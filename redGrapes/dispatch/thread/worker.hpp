/* Copyright 2020-2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <exception>
#include <thread>
#include <atomic>
#include <functional>
#include <memory>
#include <moodycamel/concurrentqueue.h>
#include <hwloc.h>
#include <redGrapes/sync/cv.hpp>
#include <redGrapes/memory/hwloc_alloc.hpp>
#include <redGrapes/memory/chunked_bump_alloc.hpp>
#include <redGrapes/task/queue.hpp>

#include <redGrapes/util/trace.hpp>
#include <redGrapes/dispatch/thread/worker_pool.hpp>

namespace redGrapes
{

namespace dispatch
{
namespace thread
{

struct WorkerThread;

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

    /*! if true, the thread shall stop
     * instead of waiting when it is out of jobs
     */
    std::atomic_bool m_stop{ false };


    std::atomic<unsigned> task_count{ 0 };

    //! condition variable for waiting if queue is empty
    CondVar cv;

    static constexpr size_t queue_capacity = 128;

public:
    memory::ChunkedBumpAlloc< memory::HwlocAlloc > & alloc;
    HwlocContext & hwloc_ctx;

    task::Queue emplacement_queue{ queue_capacity };
    task::Queue ready_queue{ queue_capacity };

    Worker( memory::ChunkedBumpAlloc< memory::HwlocAlloc > & alloc, HwlocContext & hwloc_ctx, hwloc_obj_t const & obj, WorkerId id );
    virtual ~Worker();

    inline WorkerId get_worker_id() { return id; }
    inline scheduler::WakerId get_waker_id() { return id + 1; }
    inline bool wake() { return cv.notify(); }

    virtual void stop();

    /* adds a new task to the emplacement queue
     * and wakes up thread to kickstart execution
     */
    inline void emplace_task( Task & task )
    {
        emplacement_queue.push( &task );
        wake();
    }

    inline void activate_task( Task & task )
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

    WorkerThread( memory::ChunkedBumpAlloc<memory::HwlocAlloc> & alloc, HwlocContext & hwloc_ctx, hwloc_obj_t const & obj, WorkerId worker_id );
    ~WorkerThread();

    void start();
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

