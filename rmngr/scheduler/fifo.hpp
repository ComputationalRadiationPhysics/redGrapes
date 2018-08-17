
/**
 * @file rmngr/scheduler/fifo.hpp
 */

#pragma once

#include <queue>


namespace rmngr
{

template <typename Job>
class FIFO
{
public:
    struct Property {};

    void
    push( Job const & job, Property const & prop = Property() )
    {
        std::lock_guard<std::mutex> lock( queue_mutex );
        this->queue.push( job );
    }

    bool
    empty( void )
    {
        std::lock_guard<std::mutex> lock( queue_mutex );
        return this->queue.empty();
    }

    Job
    getJob( void )
    {
        std::lock_guard<std::mutex> lock( queue_mutex );
        if ( this->queue.empty() )
            return Job(); // empty job
        else
        {
            auto job = this->queue.front();
            this->queue.pop();
            return job;
        }
    }

private:
    std::mutex queue_mutex;
    std::queue<Job> queue;
}; // struct FIFO

} // namespace rmngr
