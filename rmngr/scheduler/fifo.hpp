
/**
 * @file rmngr/scheduler/fifo.hpp
 */

#pragma once

#include <condition_variable>
#include <queue>

#include <rmngr/scheduler/dispatch.hpp>

namespace rmngr
{

template <
    typename Job
>
class FIFO
    : public DefaultJobSelector< Job >
{
public:
    struct Property {};

    FIFO()
    {}

    void finish(void)
    {
        this->cv.notify_all();
    }

    void
    push( Job const & job, Property const & prop = Property() )
    {
        {
            std::lock_guard<std::mutex> lock( queue_mutex );
            this->queue.push( job );
        }

        this->cv.notify_one();
    }

    bool
    queue_empty( void )
    {
        std::lock_guard< std::mutex >( this->queue_mutex );
        return this->queue.empty();
    }

    bool
    empty( void )
    {
        if( this->queue_empty() )
        {
            this->update();
            return this->queue_empty();
        }
        else
            return false;
    }

    Job
    getJob( void )
    {
        if( this->empty() )
        {
            std::unique_lock<std::mutex> cv_lock( this->cv_mutex );
            this->cv.wait( cv_lock );
        }

        std::lock_guard<std::mutex> lock( this->queue_mutex );
        if( this->queue.empty() )
            return Job();
        else
        {
            Job job = this->queue.front();
            this->queue.pop();

            return job;
        }
    }

private:
    std::mutex queue_mutex;
    std::queue<Job> queue;

    std::mutex cv_mutex;
    std::condition_variable cv;
}; // struct FIFO

} // namespace rmngr
