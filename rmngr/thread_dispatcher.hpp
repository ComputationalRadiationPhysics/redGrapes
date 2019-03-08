
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

/**
 * @defgroup Thread
 * @{
 * @par Required public member functions
 * - constructor: `Thread(Callable, Args&&...)` spawns a new thread which executes the callable with the given arguments
 * - `void join()`
 * @}
 */

/**
 * @defgroup Selector
 * @{
 * @par Description
 * Provides the ThreadDispatcher with Jobs.
 * A returned job must be a nullary functor returning void, i.e. of type `void _()`.
 * The public functions are required to be threadsafe.
 *
 * @par Required public member functions
 * - `bool empty()` shouldn't return true until no job will come after
 * - `Job getJob()` returns the next job. If none is available it should return a job that does nothing (or waits until new jobs are ready, if you want to avoid busy waiting)
 *
 * @}
 */

/** Manages a thread pool.
 * Worker-threads request jobs from scheduler and execute them,
 * until the ThreadDispatcher gets destroyed and all workers finished.
 *
 * @tparam Selector provides the jobs which get executed, required to implement the concept @ref Selector
 * @tparam Thread must satisfy @ref Thread
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
    std::atomic_int n_working;

  public:
    /**
     * @param selector the job selector
     * @param n_threads number of worker threads which will get spawed additionally to the main thread.
     */
    ThreadDispatcher( Selector & selector, size_t n_threads )
         : selector( selector ), n_working( 1 + n_threads  )
    {
        // set id for main thread
        thread::id = 0;

	// create worker threads
        for ( size_t i = 1; i <= n_threads; ++i )
            this->workers.emplace_back( this, i );
    }

    /**
     * Get the next job from the selector and execute it
     */
    void
    consume_job( void )
    {
        (this->selector.getJob())();
    }

    /**
     * Consume jobs until **all** threads finished (i.e. selector returns empty)
     */
    void
    work( void )
    {
        bool working = true;
        while( this->n_working > 0 )
	{
            if( this->selector.empty() )
	    {
	        if( working )
		{
		    working = false;
		    -- this->n_working;
		}
	    }
	    else
	    {
	        if( !working )
		{
		    working = true;
		    ++ this->n_working;
		}
	    }

            this->consume_job();
	}
    }

    ~ThreadDispatcher()
    {
        this->work();

        for ( Worker & worker : this->workers )
            worker.thread.join();
    }
}; // class ThreadDispatcher

} // namespace rmngr
