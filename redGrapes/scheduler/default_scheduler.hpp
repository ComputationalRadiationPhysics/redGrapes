
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
            threads.emplace_back(memory::alloc_shared< dispatch::thread::WorkerThread >( i + 1 ));
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            int rc = pthread_setaffinity_np(threads[i]->thread.native_handle(), sizeof(cpu_set_t), &cpuset);
        }

        redGrapes::dispatch::thread::current_waker_id = 0;
        
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
        SPDLOG_TRACE("free worker", id);
        //assert( id < n_workers );
        worker_state[ id / 64 ].fetch_or( (uint64_t)1 << ( id % 64 ), std::memory_order_release);
        last_free.store(id, std::memory_order_release);
    }

    /* try to allocate worker with id
     * and mark it to be busy
     *
     * @return true if worker was free and is now allocated,
     *         false if worker is already busy
     */
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
        TRACE_EVENT("Scheduler", "alloc_worker");

        for(uint64_t j = 0; j < worker_state.size(); ++j)
        {
            while( worker_state[j] > 0 )
            {
                // find index of first set bit
                // taken from https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightParallel
                uint64_t v = worker_state[j];

                unsigned int c = 64; // c will be the number of zero bits on the right
                v &= -int64_t(v);
                if (v) c--;
                if (v & 0x00000000FFFFFFFF) c -= 32;
                if (v & 0x0000FFFF0000FFFF) c -= 16;
                if (v & 0x00FF00FF00FF00FF) c -= 8;
                if (v & 0x0F0F0F0F0F0F0F0F) c -= 4;
                if (v & 0x3333333333333333) c -= 2;
                if (v & 0x5555555555555555) c -= 1;

                unsigned idx = j * 64 + c;

                if( alloc_worker( idx ) )
                    return idx;            
            }
        }

        return -1;
    }

    void activate_task( Task & task )
    {
        TRACE_EVENT("Scheduler", "activate_task");
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
        TRACE_EVENT("Scheduler", "schedule");
        SPDLOG_TRACE("schedule worker {}", worker.id);

        while( true )
        {
            Task *t = nullptr;

            // check global ready queue
            if( t = ready.pop() )
            {
                worker.queue.push(t);
                SPDLOG_TRACE("got task from ready queue");
                return true;
            }

            // try to initialize a new task
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
                // no tasks available
                free_worker( worker.id - 1 );

                if( t = ready.pop() )
                {
                    if( alloc_worker( worker.id ) )
                        worker.queue.push(t);
                    else
                        ready.push(t);

                    return true;
                }

                return false;
            }
        }
    }

    bool wake_one_worker()
    {
        TRACE_EVENT("Scheduler", "wake_one_worker");
        SPDLOG_DEBUG("DefaultScheduler: wake_one_worker()");

        int worker_id = alloc_worker();

        if( worker_id < 0 )
        {
            // FIXME: this is a workaround for a racecondition
            // a worker may be searching for a task and thus not marked free,
            // so alloc_worker will not return this worker and wake_one_worker
            // will notify no one.
            // shortly after that the worker is marked as free and begins to sleep,
            // but the newly created task will not be executed
            wake_all_workers();

            return false;
        }
        else
        {
            threads[ worker_id ]->wake();
            return true;
        }
    }

    void wake_all_workers()
    {
        for( uint16_t i = 0; i <= threads.size(); ++i )
            this->wake( i );
    }

    bool wake( WakerID id = 0 )
    {
        if( id == 0 )
            return cv.notify();
        else if( id > 0 && id <= threads.size() )
            return threads[ id - 1 ]->wake();
        else
            return false;
    }
};

} // namespace scheduler

} // namespace redGrapes

