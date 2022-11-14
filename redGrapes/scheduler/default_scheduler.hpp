
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
    task::Queue ready;
    unsigned n_workers;

    //!  bit is true if worker available, false if worker busy
    std::array< std::atomic< uint64_t >, 8> worker_state;

    std::vector<std::shared_ptr< dispatch::thread::WorkerThread >> threads;

    std::atomic< unsigned > last_free;

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

    inline void free_worker( unsigned id )
    {
        //assert( id < n_workers );
        worker_state[ id / 64 ].fetch_or( (uint64_t)1 << ( id % 64 ), std::memory_order_release);
        last_free.store(id, std::memory_order_release);
    }

    inline bool alloc_worker( unsigned id )
    {
        unsigned j = id / 64;
        unsigned k = id % 64;

        uint64_t old_val = worker_state[j].fetch_and(~((uint64_t)1 << k), std::memory_order_acquire);
        return old_val & ((uint64_t)1 << k);
    }

    /*
     * try to find an available worker
     *
     * @return worker_id if found free worker,
     *         -1 if all workers are busy
     */
    inline int alloc_worker()
    {
        unsigned off = last_free.fetch_add(1, std::memory_order_acquire) % n_workers;
        unsigned joff = off / 64;
        unsigned koff = off % 64;

        for(uint64_t j0 = 0; j0 < worker_state.size(); ++j0)
        {
            unsigned j = (j0 + joff) % worker_state.size();
            if( worker_state[j].load() != 0 )
            {
                unsigned end = 64;
                if( (j+1) == (n_workers/64) )
                    end = n_workers%64;

                for(uint64_t k0 = 0; k0 < end; ++k0)
                {
                    unsigned k = (k0 + koff) % end;

                    unsigned i = j*64 + k;
                    if( i < n_workers )
                    {
                        uint64_t old_val = worker_state[j].fetch_and(~((uint64_t)1 << k), std::memory_order_acquire);
                        if( old_val & ((uint64_t)1 << k) )
                            return i;
                    }
                    else
                        return -1;
                }
            }
        }

        return -1;
    }

    void activate_task( Task & task )
    {
        SPDLOG_TRACE("DefaultScheduler::activate_task({})", task.task_id);

        int worker_id = alloc_worker();

        if( worker_id < 0 )
            ready.push(&task);
        else
        {
            threads[ worker_id ]->queue.push(&task);
            threads[ worker_id ]->wake();
        }
    }

    // give worker a ready task if available
    // @return true if a new task was assigned to worker
    bool schedule( dispatch::thread::WorkerThread & worker )
    {
        SPDLOG_TRACE("schedule worker {}", worker.id);

        while( true )
        {
            Task *t = nullptr;

            // try to initialize a new task
            if( t = ready.pop() )
            {
                worker.queue.push(t);
                SPDLOG_TRACE("got task from ready queue");
                return true;
            }

            unsigned i = 2;
            while( i-- > 0 )
            {
                if( top_space->init_dependencies(t, true) )
                {
                    if( t )
                    {
                        // the newly initialized task is ready
                        SPDLOG_TRACE("found task in space");
                        worker.queue.push(t);
                        return true;
                    }
                }
                else
                {
                    // emplacement queue is empty
                    if( t = ready.pop() )
                    {
                        worker.queue.push(t);
                        SPDLOG_TRACE("got task from ready queue");
                        return true;
                    }

                    free_worker( worker.id );
                    return false;
                }
            }
        }
    }

    bool wake_one_worker()
    {
        SPDLOG_DEBUG("DefaultScheduler: wake_one_worker()");

        int worker_id = alloc_worker();

        if( worker_id < 0 )
            return false;
        else
        {
            threads[ worker_id ]->wake();
            return true;
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

