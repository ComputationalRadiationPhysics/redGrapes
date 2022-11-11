
#pragma once

#include <pthread.h>
#include <thread>
#include <condition_variable>
#include <bitset>

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


    unsigned n_workers;

    //!  true if worker available, false if worker busy
    std::vector<std::bitset<64>> worker_state;

    std::vector<std::shared_ptr< dispatch::thread::WorkerThread >> threads;

    DefaultScheduler( size_t n_threads = std::thread::hardware_concurrency() )
        : n_workers( n_threads )
    {
        for( size_t i = 0; i < n_threads; ++i )
        {
            threads.emplace_back(std::make_shared< dispatch::thread::WorkerThread >( i ));

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            int rc = pthread_setaffinity_np(threads[i]->thread.native_handle(), sizeof(cpu_set_t), &cpuset);

            // initially , every worker is available
            unsigned j = i / 64;
            if( j >= worker_state.size() )
                worker_state.emplace_back();

            worker_state[j].set( i % 64 );
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

    void set_worker_free( unsigned id )
    {
        worker_state[ id / 64 ].set( id % 64 );
    }
    
    /*
     * try to find an available worker
     *
     * @return worker_id if found free worker,
     *         -1 if all workers are busy
     */
    int find_free_worker()
    {
        std::lock_guard< std::mutex > lock(m);

        unsigned j = 0;
        for( std::bitset<64> & s : worker_state )
        {
            if( s.any() )
            {
                for( unsigned i = 0; i < 64; ++i )
                    if( s.test(i) )
                    {
                        s.reset(i);
                        return j * 64 + i;
                    }
            }
            j++;
        }

        return -1;
    }

    void activate_task( Task & task )
    {
        SPDLOG_TRACE("DefaultScheduler::activate_task({})", task.task_id);

        int worker_id = find_free_worker();

        if( worker_id >= 0 )
        {
            threads[ worker_id ]->queue.push(&task);
            threads[ worker_id ]->wake();
        }
        else
        {
            ready.push(&task);
        }
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
                worker.queue.push(t);
                return true;
            }

            // try to initialize a new task
            if( top_space->init_dependencies( t, true ) )
            {
                if( t )
                {
                    // the newly initialized task is ready
                    worker.queue.push(t);
                    return true;
                }
            }
            else
            {
                // emplacement queue is empty
                set_worker_free( worker.id );
                return false;
            }
        }
    }

    bool wake_one_worker()
    {
        SPDLOG_DEBUG("DefaultScheduler: wake_one_worker()");

        int worker_id = find_free_worker();
        if( worker_id >= 0 )
        {
            threads[ worker_id ]->wake();
            return true;
        }
        else
        {
            return false;
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

