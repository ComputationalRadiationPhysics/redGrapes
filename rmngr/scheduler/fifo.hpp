
/**
 * @file rmngr/scheduler/fifo.hpp
 */

#pragma once

#include <condition_variable>
#include <boost/lockfree/queue.hpp>

#include <rmngr/scheduler/dispatch.hpp>

namespace rmngr
{

/**
 * JobSelector implementation for use with DispatchPolicy.
 */
template <
    typename Job
>
class FIFO
    : public DefaultJobSelector< Job >
{
public:
    struct Property {};

    FIFO()
      : queue(0)
      , finished(false)
    {}

    // call this when no more jobs will come
    void finish()
    {
        this->finished = true;
        this->cv.notify_all();
    }

    void
    push( Job const & job, Property const & prop = Property() )
    {
        this->queue.push( job );
        this->cv.notify_one();
    }

    bool
    empty( void )
    {
        if( this->queue.empty() )
        {
            this->update();
            return this->queue.empty();
        }
        else
            return false;
    }

    Job
    getJob( void )
    {
        if( !this->finished && this->empty() )
        {
            std::unique_lock<std::mutex> cv_lock( this->cv_mutex );
            this->cv.wait( cv_lock );
        }

	Job job;
	if( this->queue.pop(job) )
            return job;
	else
            return Job();
    }

private:
    boost::lockfree::queue<Job> queue;

    std::mutex cv_mutex;
    std::condition_variable cv;

    bool finished;
}; // struct FIFO

} // namespace rmngr
