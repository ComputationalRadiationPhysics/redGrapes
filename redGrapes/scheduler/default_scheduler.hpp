
#pragma once

#include <thread>
#include <pthread.h>
#include <redGrapes/scheduler/scheduling_graph.hpp>

#include <redGrapes/imanager.hpp>
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
template<typename Task>
struct DefaultScheduler : public IScheduler
{
    std::mutex m;
    std::condition_variable cv;
    std::atomic_flag wait = ATOMIC_FLAG_INIT;

    IManager & mgr;
    std::shared_ptr< scheduler::FIFO > fifo;
    std::vector<std::shared_ptr<dispatch::thread::WorkerThread<Task>>> threads;

    DefaultScheduler( IManager & mgr, size_t n_threads = std::thread::hardware_concurrency() ) :
        mgr(mgr),
        fifo( std::make_shared< scheduler::FIFO >(mgr) )
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        sched_setaffinity (getpid(), sizeof(cpuset), &cpuset);

        for( size_t i = 0; i < n_threads; ++i )
        {
            threads.emplace_back(std::make_shared< dispatch::thread::WorkerThread<Task> >(mgr, this->fifo));

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i+1, &cpuset);
            int rc = pthread_setaffinity_np(threads[i]->thread.native_handle(),
                                            sizeof(cpu_set_t), &cpuset);
        }
        
        // if not configured otherwise,
        // the main thread will simply wait
        dispatch::thread::idle =
            [this]
            {
                SPDLOG_TRACE("DefaultScheduler::idle()");
                std::unique_lock< std::mutex > l( m );
                cv.wait( l, [this]{ return !wait.test_and_set(); } );
            };
    }

    void activate_task( TaskVertexPtr task_vertex )
    {
        fifo->activate_task( task_vertex );
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

/*! Factory function to easily create a default-scheduler object
 */
template<typename Task>
auto make_default_scheduler(IManager & mgr, size_t n_threads = std::thread::hardware_concurrency())
{
    return std::make_shared<DefaultScheduler<Task>>(mgr, n_threads);
}

} // namespace scheduler

} // namespace redGrapes

