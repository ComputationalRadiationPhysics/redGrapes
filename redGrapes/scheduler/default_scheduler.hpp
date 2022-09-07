
#pragma once

#include <pthread.h>
#include <thread>
#include <condition_variable>

#include <redGrapes/task/task_space.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/dispatch/thread/worker.hpp>

namespace redGrapes
{
namespace scheduler
{

/*
 * Combines a FIFO with worker threads
 */
struct DefaultScheduler : public IScheduler
{
    CondVar cv;

    std::mutex m;
    task::Queue ready;

    std::vector<std::shared_ptr< dispatch::thread::WorkerThread >> threads;

    DefaultScheduler( size_t n_threads = std::thread::hardware_concurrency() )
    {
        for( size_t i = 0; i < n_threads; ++i )
        {
            threads.emplace_back(std::make_shared< dispatch::thread::WorkerThread >());

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            int rc = pthread_setaffinity_np(threads[i]->thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        }
        
        // if not configured otherwise,
        // the main thread will simply wait
        redGrapes::idle =
            [this]
            {
                SPDLOG_TRACE("DefaultScheduler::idle()");
                cv.wait();
            };
    }

    void start()
    {
        for( auto & worker : threads )
            worker->start();
    }

    void stop()
    {
        for( auto & worker : threads )
            worker->stop();
    }

    void activate_task( Task & task )
    {
        task.next = nullptr;
        SPDLOG_TRACE("DefaultScheduler::activate_task({})", task.task_id);
        
        /* if one worker is idle, give it the new task */
        for( auto & worker : threads )
        {
            if( ! worker->has_work.exchange(true) )
            {
                worker->queue.push(&task);
                worker->wake();
                return;
            }
        }

        /* else add it to the ready queue */
        ready.push(&task);
    }

    // give worker a ready task if available
    void schedule( dispatch::thread::WorkerThread & worker )
    {
        if( ! worker.has_work.exchange(true) )
        {
            if( Task * t = ready.pop() )
            {
                worker.queue.push(t);
                worker.wake();
            }
            else
            {
                worker.has_work.exchange(false);
            }
        }    
    }

    // give every worker a ready task if available
    void schedule()
    {
        std::lock_guard<std::mutex> l(m);
        for( auto & worker : threads )
            schedule( *worker );
    }

    void wake_one_worker()
    {
        SPDLOG_DEBUG("DefaultScheduler: wake_one_worker()");

        for( auto & worker : threads )
        {
            if( worker->wake() )
                break;
        }
    }

    void wake_all_workers()
    {
        for( auto & worker : threads )
            worker->wake();

        this->wake();
    }

    //! wakeup sleeping main thread
    bool wake()
    {
        SPDLOG_TRACE("DefaultScheduler: wake main thread");
        return cv.notify();
    }
};

} // namespace scheduler

} // namespace redGrapes

