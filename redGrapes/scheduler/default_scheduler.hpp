
#pragma once

#include <pthread.h>
#include <thread>
#include <condition_variable>

#include <redGrapes/task/task_space.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/scheduler/fifo.hpp>
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
    std::mutex m;
    std::condition_variable cv;
    std::atomic_flag wait = ATOMIC_FLAG_INIT;

    std::shared_ptr< scheduler::FIFO > fifo;
    std::vector<std::shared_ptr< dispatch::thread::WorkerThread >> threads;

    DefaultScheduler( size_t n_threads = std::thread::hardware_concurrency() ) :
        fifo( std::make_shared< scheduler::FIFO >() )
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);

        int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        for( size_t i = 0; i < n_threads; ++i )
        {
            threads.emplace_back(std::make_shared< dispatch::thread::WorkerThread >(this->fifo));

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i+1, &cpuset);
            int rc = pthread_setaffinity_np(threads[i]->thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        }
        
        // if not configured otherwise,
        // the main thread will simply wait
        redGrapes::idle =
            [this]
            {
                SPDLOG_TRACE("DefaultScheduler::idle()");
                std::unique_lock< std::mutex > l( m );
                cv.wait( l, [this]{ return !wait.test_and_set(); } );
            };
    }

    void activate_task( Task & task )
    {
        SPDLOG_TRACE("DefaultScheduler::activate_task({})", task.task_id);
        fifo->activate_task( task );
    }

    //! wakeup sleeping worker threads
    void notify()
    {
        SPDLOG_TRACE("DefaultScheduler::notify()");
        {
            std::unique_lock< std::mutex > l( m );
            wait.clear();
        }
        cv.notify_one();

        for( auto & thread : threads )
            thread->notify();
    }
};

} // namespace scheduler

} // namespace redGrapes

