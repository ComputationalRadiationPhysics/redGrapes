
/**
 * @file rmngr/scheduler.hpp
 */

#pragma once

#include <queue>
//#include <rmngr/observer_ptr.hpp>

namespace rmngr
{

template <typename Executable>
class FIFO
{
  public:
    void
    push( Executable s )
    {
        std::lock_guard<std::mutex> lock( queue_mutex );
        this->queue.push( s );
    }

    bool
    empty( void )
    {
        std::lock_guard<std::mutex> lock( queue_mutex );
        return this->queue.empty();
    }

    Executable
    getJob( void )
    {
        Executable i;
        std::lock_guard<std::mutex> lock( queue_mutex );
        if ( this->queue.empty() )
            i = nullptr;
        else
        {
            i = this->queue.front();
            this->queue.pop();
        }
        return i;
    }

  private:
    std::mutex queue_mutex;
    std::queue<Executable> queue;
}; // struct FIFO

} // namespace rmngr
