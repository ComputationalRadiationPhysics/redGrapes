/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <deque>
#include <queue>
#include <stack>
#include <utility>
#include <thread>
#include <condition_variable>
#include <algorithm>
#include <functional>

#include <redGrapes/thread/thread_dispatcher.hpp>
#include <redGrapes/thread/thread_local.hpp>

#include <akrzemi/optional.hpp>

namespace redGrapes
{

template < typename Job >
struct ThreadSchedule
{
public:
    ThreadSchedule()
        : wakeup(false)
    {}

    void set_request_hook( std::function<void()> const & r )
    {
        request_hook = r;
    }

    void set_wait_hook( std::function<void()> const & r )
    {
        wait_hook = r;
    }

    void push( Job const & job )
    {
        queue.push( job );
        notify();
    }

    void notify()
    {
        {
            std::lock_guard< std::mutex > lock( cv_mutex );
            wakeup = true;
        }
        cv.notify_all();
    }

    void consume( std::function<bool(void)> const & pred )
    {
        wakeup = false;
        if( !pred() )
        {
            if( !queue.empty() )
            {
                Job job = queue.front();
                queue.pop();

                {
                    std::lock_guard<std::mutex> lock(stack_mutex);
                    current_jobs.push( job );
                }

                job();

                {
                    std::lock_guard<std::mutex> lock(stack_mutex);
                    current_jobs.pop();
                }
            }
            else
            {
                if( request_hook )
                    request_hook();

                if( wait_hook )
                {
                    while( ! wakeup )
                        wait_hook();
                }
                else
                {
                    std::unique_lock< std::mutex > lock( cv_mutex );
                    cv.wait(lock, [this]{ return bool(wakeup); });
                }
            }
        }
    }

    std::experimental::optional< Job >
    get_current_job()
    {
        std::lock_guard< std::mutex > lock( stack_mutex );
        if( current_jobs.empty() )
            return std::experimental::nullopt;
        else
            return current_jobs.top();
    }

private:
    std::atomic_bool wakeup;
    std::function<void()> request_hook;
    std::function<void()> wait_hook;
    std::queue< Job > queue;

    std::mutex stack_mutex;
    std::stack< Job > current_jobs;

    std::mutex cv_mutex;
    std::condition_variable cv;
}; // class ThreadSchedule

} // namespace redGrapes
