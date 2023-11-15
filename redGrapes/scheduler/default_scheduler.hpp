
#pragma once

#include <pthread.h>
#include <thread>
#include <condition_variable>

#include <redGrapes/dispatch/thread/worker.hpp>
#include <redGrapes/dispatch/thread/worker_pool.hpp>

#include <redGrapes/task/task_space.hpp>

#include <redGrapes/scheduler/scheduler.hpp>

namespace redGrapes
{
namespace scheduler
{

/*
 * Uses simple round-robin algorithm to distribute tasks to workers
 * and implements work-stealing
 */
struct DefaultScheduler : public IScheduler
{
    CondVar cv;

    DefaultScheduler();

   void idle();

    /* send the new task to a worker
     */
    void emplace_task( Task & task );

    /* send this already existing,
     * but only through follower-list so it is not assigned to a worker yet.
     * since this task is now ready, send find a worker for it
     */
    void activate_task( Task & task );

    /* tries to find a task with uninialized dependency edges in the
     * task-graph in the emplacement queues of other workers
     * and removes it from there
     */
    Task * steal_new_task( dispatch::thread::Worker & worker );

    /* tries to find a ready task in any queue of other workers
     * and removes it from the queue
     */
    Task * steal_ready_task( dispatch::thread::Worker & worker );

    // give worker a ready task if available
    // @return task if a new task was found, nullptr otherwise
    Task * steal_task( dispatch::thread::Worker & worker );

    /* Wakeup some worker or the main thread
     *
     * WakerId = 0 for main thread
     * WakerId = WorkerId + 1
     *
     * @return true if thread was indeed asleep
     */
    bool wake( WakerId id = 0 );

    /* wakeup all wakers (workers + main thread)
     */
    void wake_all();
};

} // namespace scheduler

} // namespace redGrapes

