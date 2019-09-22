
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
        : job_request(false)
    {}

    void set_request_hook( std::function<void()> const & r )
    {
        request_hook = r;
    }

    void notify()
    {
        cv.notify_all();
    }

    bool needs_job()
    {
        return job_request && queue.empty();
    }

    void push( Job const & job )
    {
        queue.push( job );
        job_request = false;

        notify();
    }

    void consume( std::function<bool(void)> const & pred )
    {
        {
            std::unique_lock< std::mutex > lock( cv_mutex );
            cv.wait(
                lock,
                [this, pred, &lock]
                {
                    if( request_hook && queue.empty() )
                    {
                        job_request = true;

                        lock.unlock();
                        request_hook();
                        lock.lock();
                    }

                    return pred() || !queue.empty();
                }
            );
        }

        if( !queue.empty() && !pred() )
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
    bool job_request;
    std::function<void()> request_hook;
    std::queue< Job > queue;

    std::mutex stack_mutex;
    std::stack< Job > current_jobs;

    std::mutex cv_mutex;
    std::condition_variable cv;
}; // class ThreadSchedule

} // namespace rmngr

