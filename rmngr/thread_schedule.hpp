
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

    bool empty()
    {
        std::lock_guard< std::mutex > lock( mutex );
        return queue.empty() && current_jobs.empty();
    }

    bool needs_job()
    {
        std::lock_guard< std::mutex > lock( mutex );
        return queue.empty() && (current_jobs.empty() || job_request);
    }

    void push( Job const & job )
    {
        {
            std::lock_guard< std::mutex > lock( mutex );
            //std::cerr << "thread schedule: added job"<<std::endl;
            queue.push( job );
            job_request = false;
        }
        notify();
    }

    void consume( std::function<bool(void)> const & pred )
    {
        std::unique_lock< std::mutex > lock( mutex );
        job_request = true;
        /*
        lock.unlock();
        if(needs_job())
            request_hook();
        lock.lock();
        */
        //std::cerr << "thread["<<thread::id<<"] REQUEST JOB"<<std::endl;

        cv.wait(
            lock,
            [this, pred]{ return pred() || !queue.empty(); }
        );
        job_request = false;
        //std::cerr << "thread["<<thread::id<<"] notified" <<std::endl;

        if( !queue.empty() )
        {
            //std::cout << "thread["<<thread::id<<"] HAS JOB"<<std::endl;
            Job job = queue.front();
            queue.pop();

            current_jobs.push( job );
            lock.unlock();

            //std::cerr << "threadschedule: begin job"<<std::endl;
            job();
            //std::cerr << "threadschedule: end job"<<std::endl;
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
    std::function<void()> request_hook;
    std::mutex mutex;
    std::queue< Job > queue;
    std::stack< Job > current_jobs;
    std::condition_variable cv;

    bool job_request;
}; // class ThreadSchedule

} // namespace rmngr

