
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
    void notify()
    {
        cv.notify_all();
    }

    bool empty()
    {
        std::lock_guard< std::mutex > lock( mutex );
        return queue.empty() && current_jobs.empty();
    }

    void push( Job const & job )
    {
        {
            std::lock_guard< std::mutex > lock( mutex );
            queue.push( job );
        }
        notify();
    }

    void consume( std::function<bool(void)> const & pred )
    {
        std::unique_lock< std::mutex > lock( mutex );
        cv.wait(
            lock,
            [this, pred]{ return pred() || !queue.empty(); }
        );

        if( !queue.empty() )
        {
            Job job = queue.front();
            queue.pop();

            current_jobs.push( job );
            lock.unlock();
            job();
            lock.lock();
            current_jobs.pop();
        }
    }

    std::experimental::optional< Job >
    get_current_job()
    {
        std::lock_guard< std::mutex > lock( mutex );
        if( current_jobs.empty() )
            return std::experimental::nullopt;
        else
            return current_jobs.top();
    }

private:
    std::mutex mutex;
    std::queue< Job > queue;
    std::stack< Job > current_jobs;
    std::condition_variable cv;
}; // class ThreadSchedule

} // namespace rmngr

