
/**
 * @file rmngr/thread_dispatcher.hpp
 */

#pragma once

#include <atomic>
#include <thread>
#include <vector>

namespace rmngr
{

namespace thread
{
static thread_local size_t id;
}

/** Manages a thread pool.
 * Worker-threads request jobs from scheduler and execute them,
 * until the ThreadDispatcher gets destroyed and all workers finished.
 *
 * @tparam Scheduler needs lockfree Callable getJob(int id) and bool empty()
 * @tparam Thread must create a thread on construction and have join().
 */
template <typename Selector, typename Thread = std::thread>
class ThreadDispatcher
{
  private:
    struct Worker
    {
        Worker( ThreadDispatcher * td, size_t id )
            : thread( work, td, this, id )
        {
        }
        Worker( Worker && w ) : thread( std::move( w.thread ) ) {}

        static void
        work( ThreadDispatcher * td, Worker * worker, size_t id )
        {
            thread::id = id;
            td->work();
        }

        Thread thread;
    }; // struct Worker

    Selector & selector;
    std::vector<Worker> workers;
    std::atomic_bool running;

  public:
    ThreadDispatcher( Selector & selector_, size_t n_threads )
        : selector( selector_ ), running( true )
    {
        thread::id = 0;
        for ( size_t i = 1; i <= n_threads; ++i )
            this->workers.emplace_back( this, i );
    }

    void
    consume_job( void )
    {
        (this->selector.getJob())();
    }

    void
    work( void )
    {
        while( ! this->selector.empty() )
            this->consume_job();

        this->selector.finish();
    }

    ~ThreadDispatcher()
    {
        this->work();

        for ( Worker & worker : this->workers )
            worker.thread.join();
    }
}; // class ThreadDispatcher

} // namespace rmngr
