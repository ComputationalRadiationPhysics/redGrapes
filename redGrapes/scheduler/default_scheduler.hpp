
#pragma once

#include <pthread.h>
#include <thread>
#include <condition_variable>

#include <redGrapes/dispatch/thread/worker.hpp>
#include <redGrapes/dispatch/thread/worker_pool.hpp>

#include <redGrapes/context.hpp>
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

    DefaultScheduler()
    {
        // if not configured otherwise,
        // the main thread will simply wait
        redGrapes::idle =
            [this]
            {
                SPDLOG_TRACE("DefaultScheduler::idle()");
                cv.wait();
            };
    }

    /* send the new task to a worker
     */
    void emplace_task( Task & task )
    {
        // todo: properly store affinity information in task
        dispatch::thread::WorkerId worker_id = task->arena_id % worker_pool->size();

        worker_pool->get_worker(worker_id).emplace_task( &task );
    }

    /* send this already existing,
     * but only through follower-list so it is not assigned to a worker yet.
     * since this task is now ready, send find a worker for it
     */
    void activate_task( Task & task )
    {
        //! worker id to use in case all workers are busy
        static thread_local std::atomic< unsigned int > next_worker(dispatch::thread::current_worker ?
                                                                    dispatch::thread::current_worker->get_worker_id() + 1 : 0);

        TRACE_EVENT("Scheduler", "activate_task");
        SPDLOG_TRACE("DefaultScheduler::activate_task({})", task.task_id);

        int worker_id = worker_pool->find_free_worker();
        if( worker_id < 0 )
        {
            worker_id = next_worker.fetch_add(1) % worker_pool->size();
            if( worker_id == dispatch::thread::current_worker->get_worker_id() )
                worker_id = next_worker.fetch_add(1) % worker_pool->size();
        }

        worker_pool->get_worker( worker_id ).ready_queue.push(&task);
        worker_pool->set_worker_state( worker_id, dispatch::thread::WorkerState::BUSY );
        worker_pool->get_worker( worker_id ).wake();
    }
    
    /* tries to find a task with uninialized dependency edges in the
     * task-graph in the emplacement queues of other workers
     * and removes it from there
     */
    Task * steal_new_task( dispatch::thread::WorkerThread & worker )
    {
        std::optional<Task*> task = worker_pool->probe_worker_by_state<Task*>(
            [&worker](unsigned idx) -> std::optional<Task*>
            {
                // we have a candidate of a busy worker,
                // now check its queue
                if(Task* t = worker_pool->get_worker(idx).emplacement_queue.pop())
                    return t;

                // otherwise check own queue again
                else if(Task* t = worker.emplacement_queue.pop())
                    return t;

                // else continue search
                else
                    return std::nullopt;
            },

            // find a busy worker
            dispatch::thread::WorkerState::BUSY,

            // start next to current worker
            worker.get_worker_id());

        return task ? *task : nullptr;
    }

    /* tries to find a ready task in any queue of other workers
     * and removes it from the queue
     */
    Task * steal_ready_task( dispatch::thread::WorkerThread & worker )
    {
        std::optional<Task*> task = worker_pool->probe_worker_by_state<Task*>(
            [&worker](unsigned idx) -> std::optional<Task*>
            {
                // we have a candidate of a busy worker,
                // now check its queue
                if(Task* t = worker_pool->get_worker(idx).ready_queue.pop())
                    return t;

                // otherwise check own queue again
                else if(Task* t = worker.ready_queue.pop())
                    return t;

                // else continue search
                else
                    return std::nullopt;
            },

            // find a busy worker
            dispatch::thread::WorkerState::BUSY,

            // start next to current worker
            worker.get_worker_id());

        return task ? *task : nullptr;
    }

    // give worker a ready task if available
    // @return task if a new task was found, nullptr otherwise
    Task * steal_task( dispatch::thread::WorkerThread & worker )
    {
        unsigned worker_id = worker.get_worker_id();

        SPDLOG_INFO("steal task for worker {}", worker_id);

        if( Task * task = steal_ready_task( worker ) )
        {
            worker_pool->set_worker_state( worker_id, dispatch::thread::WorkerState::BUSY );
            return task;
        }

        if( Task * task = steal_new_task( worker ) )
        {
            task->pre_event.up();
            task->init_graph();

            if( task->get_pre_event().notify( true ) )
            {
                worker_pool->set_worker_state( worker_id, dispatch::thread::WorkerState::BUSY );
                return task;
            }            
        }

        return nullptr;
    }

    /* Wakeup some worker or the main thread
     *
     * WakerId = 0 for main thread
     * WakerId = WorkerId + 1
     *
     * @return true if thread was indeed asleep
     */
    bool wake( WakerId id = 0 )
    {
        if( id == 0 )
            return cv.notify();
        else if( id > 0 && id <= worker_pool->size() )
            return worker_pool->get_worker(id - 1).wake();
        else
            return false;
    }

    /* wakeup all wakers (workers + main thread)
     */
    void wake_all()
    {
        for( uint16_t i = 0; i <= worker_pool->size(); ++i )
            this->wake( i );
    }
};

} // namespace scheduler

} // namespace redGrapes

