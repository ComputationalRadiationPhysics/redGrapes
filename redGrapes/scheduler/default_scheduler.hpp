
#pragma once

#include <thread>

#include <redGrapes/graph/scheduling_graph.hpp>

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
template <
    typename TaskID,
    typename TaskPtr
>
struct DefaultScheduler : public IScheduler< TaskPtr >
{
    using EventID = typename redGrapes::SchedulingGraph< TaskID, TaskPtr >::EventID;

    redGrapes::SchedulingGraph< TaskID, TaskPtr > & scheduling_graph;
    std::shared_ptr< redGrapes::scheduler::FIFO< TaskID, TaskPtr > > fifo;
    std::vector< std::shared_ptr< redGrapes::scheduler::WorkerThread<> > > threads;
    redGrapes::scheduler::DefaultWorker main_thread_worker;

    //  main thread idle
    std::mutex m;
    std::condition_variable cv;
    std::atomic_flag wait = ATOMIC_FLAG_INIT;

    DefaultScheduler(
        redGrapes::SchedulingGraph< TaskID, TaskPtr > & scheduling_graph,
        std::function< bool ( TaskPtr ) > mgr_run_task,
        std::function< void ( TaskPtr ) > mgr_activate_followers,
        std::function< void ( TaskPtr ) > mgr_remove_task
    ) :
        scheduling_graph( scheduling_graph ),
        fifo(
            std::make_shared<
                redGrapes::scheduler::FIFO< TaskID, TaskPtr >
            >(
                scheduling_graph,
                mgr_run_task,
                mgr_activate_followers,
                mgr_remove_task
            )
        ),
        main_thread_worker( [this]{ return false; } )
    {
        // spawn worker threads
        int n_threads = std::thread::hardware_concurrency();
        for( int i = 0; i < n_threads; ++i )
            threads.emplace_back(
                 std::make_shared< redGrapes::scheduler::WorkerThread<> >(
                     [this] { return this->fifo->consume(); }
                 )
            );

        thread::idle =
            [this]
            {
                std::unique_lock< std::mutex > l( m );
                cv.wait( l, [this]{ return !wait.test_and_set(); } );
            };
    }

    //! wakeup sleeping worker threads
    void notify()
    {
        {
            std::unique_lock< std::mutex > l( m );
            wait.clear();
        }
        cv.notify_one();

        for( auto & thread : threads )
            thread->worker.notify();
    }

    void
    activate_task( TaskPtr task_ptr )
    {
        fifo->activate_task( task_ptr );
        notify();
    }
};

} // namespace scheduler

} // namespace redGrapes

