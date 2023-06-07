
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
    unsigned n_workers;

    //! bit is true if worker available, false if worker busy
    std::array< std::atomic< uint64_t >, 8> worker_state;
    std::vector<std::shared_ptr< dispatch::thread::WorkerThread >> threads;

    //! worker id to use in case all workers are busy
    std::atomic< unsigned > wid;

    DefaultScheduler( size_t n_threads = std::thread::hardware_concurrency() )
        : n_workers( n_threads )
        , wid(0)
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

    /* set worker state to signal it has no tasks
     */
    inline void free_worker( unsigned id )
    {
        SPDLOG_TRACE("free worker", id);
        //assert( id < n_workers );
        worker_state[ id / 64 ].fetch_or( (uint64_t)1 << ( id % 64 ), std::memory_order_release);
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

    // find index of first set bit
    // taken from https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightParallel
    unsigned int first_one_idx( uint64_t v )
    {
        unsigned int c = 64; // c will be the number of zero bits on the right
        v &= -int64_t(v);
        if (v) c--;
        if (v & 0x00000000FFFFFFFF) c -= 32;
        if (v & 0x0000FFFF0000FFFF) c -= 16;
        if (v & 0x00FF00FF00FF00FF) c -= 8;
        if (v & 0x0F0F0F0F0F0F0F0F) c -= 4;
        if (v & 0x3333333333333333) c -= 2;
        if (v & 0x5555555555555555) c -= 1;

        return c;
    }
    
    /*
     * try to find an available worker,
     * returns a busy worker if no free worker is available
     * @return worker_id
     */
    inline int find_worker()
    {
        TRACE_EVENT("Scheduler", "find_worker");

        SPDLOG_TRACE("find worker...");
       
        for(uint64_t j = 0; j < (1 + (n_workers-1)/64); ++j)
        {
            uint64_t mask = -1;
            if( j == n_workers/64 )
                mask = (1 << (n_workers%64)) - 1;

            while( worker_state[j]&mask > 0 )
            {
                unsigned int c = first_one_idx( worker_state[j]&mask );

                if( c < 64 )
                {
                    unsigned int idx = j * 64 + c;

                    if( alloc_worker( idx ) )
                        return idx;
                }
            }
        }

        SPDLOG_TRACE("no free worker found");

        // no free worker found,
        return -1;
        // return a busy worker instead.
    }

    void activate_task( Task & task )
    {
        TRACE_EVENT("Scheduler", "activate_task");
        SPDLOG_TRACE("DefaultScheduler::activate_task({})", task.task_id);

        int worker_id = find_worker();
        if( worker_id < 0 )
        {
            worker_id = wid.fetch_add(1) % n_workers;
            alloc_worker(worker_id);
        }

        //spdlog::info("activate to worker {}", worker_id);
        threads[ worker_id ]->queue->push(&task);
        threads[ worker_id ]->wake();
    }

    /* tries to find a ready task in any queue of other workers
     * and removes it from the queue
     */
    Task * steal_task( )
    {
        spdlog::info("steal task");
        for(uint64_t j = 0; j < (1 + (n_workers-1)/64); ++j)
        {
            uint64_t mask = -1;
            if( j == n_workers/64 )
                mask = ((uint64_t)1 << (n_workers%64)) - 1;

            while( ((~worker_state[j])&mask) > 0 )
            {
                // find index of first *busy* worker,
                // hence invert worker state

                unsigned int k = first_one_idx((~worker_state[j])&mask);
                if( k < 64 ) {
                    unsigned int idx = j * 64 + k;

                    spdlog::info("try idx {}", idx);

                    if( Task * t = threads[ idx ]->queue->pop() )
                    {
                        spdlog::info("stole task");
                        return t;
                    }

                    // dont check this worker again
                    mask &= ~((uint64_t)1 << k);
                }
            }
        }

        spdlog::info("found no task");

        return nullptr;        
    }
    
    // give worker a ready task if available
    // @return true if a new task was assigned to worker
    bool schedule( dispatch::thread::WorkerThread & worker )
    {
        TRACE_EVENT("Scheduler", "schedule");
        SPDLOG_INFO("schedule worker {}", worker.id);

        Task * task = nullptr;

        /*
        if( task = steal_task() )
        {
            worker.queue->push( task );
            return true;
        }
        */

        unsigned worker_id = worker.get_worker_id();
        free_worker( worker_id );
        SPDLOG_DEBUG("free worker {}", worker_id);

        // while worker is still available
        while( worker_state[ worker_id/64 ] & ((uint64_t)1 << (worker_id%64)) )
        {
            // try to initialize a new task
            if( top_space->init_dependencies(task, true) )
            {
                if( task )
                {
                    // the newly initialized task is ready
                    SPDLOG_DEBUG("worker {} found new ready task in space", worker_id);
                    worker.queue->push( task );
                    alloc_worker( worker_id );
                    return true;
                }
            }
            else
            {
                // no tasks available
                return false;
            }
        }

        SPDLOG_DEBUG("worker {} got activated while searching", worker_id);

        // worker got activated from event
        return true;
    }

    bool wake_one_worker()
    {
        TRACE_EVENT("Scheduler", "wake_one_worker");
        SPDLOG_INFO("DefaultScheduler: wake_one_worker()");

        int worker_id = find_worker();
        if( worker_id < 0 )
        {
            // FIXME: this is a workaround for a racecondition
            // a worker may be searching for a task and thus not marked free,
            // so alloc_worker will not return this worker and wake_one_worker
            // will notify no one.
            // shortly after that the worker is marked as free and begins to sleep,
            // but the newly created task will not be executed
            SPDLOG_INFO("no busy worker found, wake all");

            //wake_all_workers();
            return false;
        }
        else
        {
            SPDLOG_INFO("wake worker {}", worker_id);
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

