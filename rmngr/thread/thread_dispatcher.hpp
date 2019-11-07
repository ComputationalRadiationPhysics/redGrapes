/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file rmngr/thread_dispatcher.hpp
 */

#pragma once

#include <thread>

namespace rmngr
{

namespace thread
{
static thread_local std::size_t id;
static thread_local unsigned int scope_level;
}

/**
 * @defgroup Thread
 * @{
 * @par Required public member functions
 * - constructor: `Thread(Callable, Args&&...)` spawns a new thread which executes the callable with the given arguments
 * - `void join()`
 * @}
 */

/** Manages a thread pool.
 * Worker-threads request jobs from scheduler and execute them,
 * until the ThreadDispatcher gets destroyed and all workers finished.
 *
 * @tparam JobSelector must implement `bool empty()` and `void consume_job()`
 * @tparam Thread must satisfy @ref Thread
 */    
template <
    typename JobSelector,
    typename Thread = std::thread
>
class ThreadDispatcher
{
public:
    ThreadDispatcher( JobSelector & job_selector, std::size_t n_threads = 1 )
        : job_selector( job_selector )
    {
        // set id for main thread
        thread::id = 0;

	// create worker threads
        for ( std::size_t i = 1; i < n_threads; ++i )
            this->threads.emplace_back(
                [this, i]
                {
                    thread::id = i;
                    this->work();
                }
            );
    }

    void finish()
    {
        this->work();

        for ( Thread & thread : this->threads )
            thread.join();
    }

private:
    void work()
    {
        while( ! job_selector.empty() )
            job_selector.consume_job();
    }

    JobSelector & job_selector;
    std::vector< Thread > threads;
}; // class ThreadDispatcher

} // namespace rmngr
