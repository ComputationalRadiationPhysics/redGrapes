
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
            CPU_SET(2*i, &cpuset);
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
    // @return true if a new task was assigned to worker
    bool schedule( dispatch::thread::WorkerThread & worker )
    {
        SPDLOG_TRACE("schedule worker {}", (void*)&worker);
        while( true )
        {
            Task *t = nullptr;

            if( t = ready.pop() )
            {
                worker.has_work.exchange(true);
                worker.queue.push(t);
                return true;
            }

            // try to initialize a new task
            if( top_space->init_dependencies( t, true ) )
            {
                if( t )
                {
                    // the newly initialized task is ready
                    worker.has_work.exchange(true);
                    worker.queue.push(t);
                    return true;
                }
            }
            else
                // emplacement queue is empty
                return false;

        }
    }

    bool wake_one_worker()
    {
        SPDLOG_DEBUG("DefaultScheduler: wake_one_worker()");

        for( auto & worker : threads )
        {
            if( worker->wake() )
                return true;
        }

        return false;
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

