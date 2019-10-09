/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <deque>
#include <queue>
#include <stack>
#include <utility>
#include <thread>
#include <condition_variable>
#include <algorithm>
#include <functional>

#include <akrzemi/optional.hpp>

namespace rmngr
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

                std::unique_lock< std::mutex > lock( cv_mutex );
                cv.wait(lock, [this]{ return wakeup; });
                wakeup = false;
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
    bool wakeup;
    std::function<void()> request_hook;
    std::queue< Job > queue;

    std::mutex stack_mutex;
    std::stack< Job > current_jobs;

    std::mutex cv_mutex;
    std::condition_variable cv;
}; // class ThreadSchedule

} // namespace rmngr done
