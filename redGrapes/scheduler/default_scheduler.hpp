
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

    void activate_task( Task & task )
    {
        SPDLOG_TRACE("DefaultScheduler::activate_task({})", task.task_id);
        ready.push(&task);
    }

    void schedule()
    {
        SPDLOG_INFO("schedule");
        // give every empty worker a ready task

        std::lock_guard<std::mutex> l(m);

        for( auto worker : threads )
        {
            if( worker->queue.empty() )
            {
                Task * t = ready.pop();
                if(t)
                {
                    worker->queue.push(t);
                    worker->wake();
                }
            }
        }
    }

    void wake_one_worker()
    {
        SPDLOG_TRACE("DefaultScheduler: wake_one_worker()");

        for( auto worker : threads )
        {
            if( worker->wake() )
                break;
        }
    }

    void wake_all_workers()
    {
        for( auto worker : threads )
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

