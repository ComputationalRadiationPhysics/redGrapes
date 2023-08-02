
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
    static constexpr uint64_t bitfield_len = 64;
    std::vector< std::atomic< uint64_t > > worker_state;
    std::vector<std::shared_ptr< dispatch::thread::WorkerThread >> threads;

    DefaultScheduler( size_t n_threads = std::thread::hardware_concurrency() )
        : n_workers( n_threads )
        , worker_state( ceil_div(n_threads, bitfield_len)  )
    {
        threads.reserve( n_threads );

        unsigned n_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
        if( n_threads > n_pus )
            spdlog::warn("{} worker-threads requested, but only {} PUs available!", n_threads, n_pus);

        for( size_t i = 0; i < n_threads; ++i )
        {
            auto worker = memory::alloc_shared_bind< dispatch::thread::WorkerThread >( i, i );
            threads.emplace_back( worker );
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

    /* signals all workers to start executing tasks
     */
    void start()
    {
        bool all_ready = false;
        while( ! all_ready )
        {
            all_ready = true;
            for( auto & worker : threads )
                all_ready &= worker->ready.load();
        }
        
        for( auto & worker : threads )
            worker->start();
    }

    /* signals all workers that no new tasks will be added
     */
    void stop()
    {
        for( auto & worker : threads )
            worker->stop();
    }

    inline bool is_worker_free( unsigned worker_id )
    {
        return worker_state[ worker_id/64 ] & ((uint64_t)1 << (worker_id%64));
    }
    inline bool is_worker_busy( unsigned worker_id )
    {
        return !is_worker_free( worker_id );
    }

    /* sets the worker-state bitfield to free for the specified worker
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
    inline unsigned int first_one_idx( uint64_t v )
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

    /*! calculates  ceil( a / b )
     */
    inline uint64_t ceil_div( uint64_t a, uint64_t b )
    {
        return (a+b-1)/b;
    }


    template <typename T, typename F>
    inline std::optional< T > find_worker_in_field( unsigned j, uint64_t mask, bool expected_worker_state, F && f )
    {
        while( true )
        {
            uint64_t field = expected_worker_state ?
                uint64_t(worker_state[j]) : ~uint64_t(worker_state[j]);

            uint64_t masked_field = field & mask;
            if( masked_field == 0 )
                break;

            // find index of first worker
            unsigned int k = first_one_idx( masked_field );

            if( k < bitfield_len )
            {
                unsigned int idx = j * bitfield_len + k;
                //spdlog::info("find worker: j = {}, k = {}, idx= {}", j , k, idx);

                if( std::optional<T> x = f( idx ) )
                    return x;

                // dont check this worker again
                mask &= ~(uint64_t(1) << k);
            }
        }

        return std::nullopt;
    }
    
    template <typename T, typename F>
    inline std::optional< T >
    find_worker(
        F && f,
        bool expected_worker_state,
        unsigned start_worker_idx,
        bool exclude_start = true)
    {
        uint64_t start_field_idx = start_worker_idx / bitfield_len;
        uint64_t first_mask = (uint64_t(1) << (start_worker_idx%bitfield_len)) - 1;

        {
            uint64_t mask = ~first_mask;
            if( start_field_idx == worker_state.size() - 1 && n_workers % bitfield_len != 0 )
                mask &= (uint64_t(1) << (n_workers % bitfield_len)) - 1;

            if( exclude_start )
                mask &= ~(uint64_t(1) << (start_worker_idx%bitfield_len));

            if( auto x = find_worker_in_field<T>( start_field_idx, mask, expected_worker_state, f ) )
                return x;
        }

        for(
            uint64_t b = 1;
            b < ceil_div(n_workers, bitfield_len);
            ++b
        ) {
            uint64_t field_idx = (start_field_idx + b) % worker_state.size();
            uint64_t mask = ~0;

            if( field_idx == worker_state.size() - 1 && n_workers % bitfield_len != 0 )
                mask &= (uint64_t(1) << (n_workers % bitfield_len)) - 1;

            if( auto x = find_worker_in_field<T>( field_idx, mask, expected_worker_state, f ) )
                return x;
        }

        {
            uint64_t mask = first_mask;
            if( start_field_idx == worker_state.size() - 1 && n_workers % bitfield_len != 0 )
                mask &= (uint64_t(1) << (n_workers % bitfield_len)) - 1;
            if( auto x = find_worker_in_field<T>( start_field_idx, mask, expected_worker_state, f ) )
                return x;
        }

        return std::nullopt;
    }

    /*
     * try to find an available worker,
     * returns a busy worker if no free worker is available
     * @return worker_id
     */
    inline int find_free_worker()
    {
        TRACE_EVENT("Scheduler", "find_worker");

        SPDLOG_TRACE("find worker...");

        unsigned start_idx = 0;
        if( auto w = dispatch::thread::current_worker )
            start_idx = w->get_worker_id();

        std::optional<unsigned> idx = find_worker<unsigned>([this]( unsigned idx ) -> std::optional<unsigned> {
                                                                if( alloc_worker( idx ) )
                                                                    return idx;
                                                                else
                                                                    return std::nullopt;
                                                            },
                                                            true, // find a free worker
                                                            start_idx,
                                                            false);

        if( idx )
            return *idx;
        else
            // no free worker found,
            return -1;
    }

    /* tries to find a ready task in any queue of other workers
     * and removes it from the queue
     */
    Task * steal_new_task( dispatch::thread::WorkerThread & worker )
    {
        std::optional<Task*> task = find_worker<Task*>(
                                                       [this, &worker]( unsigned idx ) -> std::optional<Task*> {
                if( Task * t = this->threads[ idx ]->emplacement_queue->pop() )
                    return t;
                else if( Task * t = worker.emplacement_queue->pop() )
                    return t;
                else
                    return std::nullopt;
            },
            false, // find a busy worker
            worker.get_worker_id());

        if( task )
            return *task;
        else
            return nullptr;
    }

    /* tries to find a ready task in any queue of other workers
     * and removes it from the queue
     */
    Task * steal_ready_task( dispatch::thread::WorkerThread & worker )
    {
        std::optional<Task*> task = find_worker<Task*>(
                                                       [this, &worker]( unsigned idx ) -> std::optional<Task*> {
                                                           if( Task * t = this->threads[ idx ]->ready_queue->pop() )
                    return t;
                                                           else if( Task * t = worker.ready_queue->pop() )
                                                               return t;

                else
                    return std::nullopt;
            },
            false, // find a busy worker
            worker.get_worker_id());

        if( task )
            return *task;
        else
            return nullptr;
    }

    // give worker a ready task if available
    // @return task if a new task was found, nullptr otherwise
    Task * schedule( dispatch::thread::WorkerThread & worker )
    {
        unsigned worker_id = worker.get_worker_id();

        TRACE_EVENT("Scheduler", "schedule");
        SPDLOG_INFO("schedule worker {}", worker_id);
        
        Task * task = nullptr;

        while( worker.init_dependencies( task, true ) )
        {
            if( task )
                return task;
        }

        free_worker( worker_id );

        if( task = steal_ready_task( worker ) )
        {
            alloc_worker( worker_id );
            return task;
        }

        if( task = steal_new_task( worker ) )
        {
            task->pre_event.up();
            task->init_graph();

            if( task->get_pre_event().notify( true ) )
            {
                alloc_worker( worker_id );
                return task;
            }            
        }

        return nullptr;
    }

    void emplace_task( Task & task )
    {
        static std::atomic< unsigned int > next_worker(0);
        auto & worker = threads[ next_worker.fetch_add(1) % n_workers ];
        worker->emplace_task( &task );
    }

    void activate_task( Task & task )
    {
        //! worker id to use in case all workers are busy
        static thread_local std::atomic< unsigned int > next_worker( dispatch::thread::current_worker->get_worker_id() + 1 );

        TRACE_EVENT("Scheduler", "activate_task");
        SPDLOG_TRACE("DefaultScheduler::activate_task({})", task.task_id);

        int worker_id = find_free_worker();
        if( worker_id < 0 )
        {
            worker_id = next_worker.fetch_add(1) % n_workers;
            if( worker_id == dispatch::thread::current_worker->get_worker_id() )
                worker_id = next_worker.fetch_add(1) % n_workers;
        }

        threads[ worker_id ]->ready_queue->push(&task);
        alloc_worker(worker_id);

        threads[ worker_id ]->wake();
    }

    bool wake_one_worker()
    {
        TRACE_EVENT("Scheduler", "wake_one_worker");
        SPDLOG_INFO("DefaultScheduler: wake_one_worker()");

        int worker_id = find_free_worker();
        if( worker_id < 0 )
        {
            // FIXME: this is a workaround for a racecondition
            // a worker may be searching for a task and thus not marked free,
            // so alloc_worker will not return this worker and wake_one_worker
            // will notify no one.
            // shortly after that the worker is marked as free and begins to sleep,
            // but the newly created task will not be executed
            SPDLOG_INFO("no busy worker found, wake all");

            wake_all_workers();
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

