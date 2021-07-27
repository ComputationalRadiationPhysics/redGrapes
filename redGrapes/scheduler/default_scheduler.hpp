
#pragma once

#include <thread>

#include <redGrapes/graph/scheduling_graph.hpp>

#include <redGrapes/imanager.hpp>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/scheduler/scheduler.hpp>
#include <redGrapes/scheduler/fifo.hpp>
#include <redGrapes/scheduler/worker.hpp>

namespace redGrapes
{
namespace scheduler
{

/*
 * Combines a FIFO with worker threads
 */
template<typename Task>
struct DefaultScheduler : public IScheduler<Task>
{
    std::mutex m;
    std::condition_variable cv;
    std::atomic_flag wait = ATOMIC_FLAG_INIT;

    IManager<Task> & mgr;
    std::shared_ptr<redGrapes::scheduler::FIFO<Task>> fifo;
    std::vector<std::shared_ptr<redGrapes::scheduler::WorkerThread<>>> threads;
    
    DefaultScheduler( IManager<Task> & mgr, size_t n_threads = std::thread::hardware_concurrency() ) :
        mgr(mgr),
        fifo( std::make_shared< redGrapes::scheduler::FIFO< Task > >(mgr) )
    {
        for( size_t i = 0; i < n_threads; ++i )
            threads.emplace_back(
                 std::make_shared< redGrapes::scheduler::WorkerThread<> >(
                     [this] { return this->fifo->consume(); }
                 )
            );

        // if not configured otherwise,
        // the main thread will simply wait
        thread::idle =
            [this]
            {
                spdlog::trace("DefaultScheduler::idle()");
                std::unique_lock< std::mutex > l( m );
                cv.wait( l, [this]{ return !wait.test_and_set(); } );
            };
    }

    //! wakeup sleeping worker threads
    void notify()
    {
        spdlog::trace("DefaultScheduler::notify()");
        {
            std::unique_lock< std::mutex > l( m );
            wait.clear();
        }
        cv.notify_one();

        for( auto & thread : threads )
            thread->worker.notify();
    }

    bool
    activate_task( std::shared_ptr<PrecedenceGraphVertex<Task>> task_vertex_ptr )
    {
        return fifo->activate_task( task_vertex_ptr );
    }
};

/*! Factory function to easily create a default-scheduler object
 */
template<typename Task>
auto make_default_scheduler(IManager<Task>& mgr, size_t n_threads = std::thread::hardware_concurrency())
{
    return std::make_shared<DefaultScheduler<Task>>(mgr, n_threads);
}

} // namespace scheduler

} // namespace redGrapes

