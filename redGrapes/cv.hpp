
#pragma once

#include <condition_variable>

namespace redGrapes
{

struct CondVar
{
    std::condition_variable cv;
    std::mutex m;
    std::atomic_flag wait_flag = ATOMIC_FLAG_INIT;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;

    volatile unsigned count;
    unsigned limit;

    CondVar()
        : count(0)
        , limit(100000)
    {
    }
    
    void wait()
    {
        while( wait_flag.test_and_set(std::memory_order_acquire) )
            if( ++count > limit )
            {
                busy.clear();

                std::unique_lock< std::mutex > l( m );
                cv.wait( l, [this]{ return !wait_flag.test_and_set(std::memory_order_acquire); } );
                l.unlock();

                count = 0;
                return;
            }
    }

    void notify()
    {
        if( ! busy.test_and_set() )
        {
            std::unique_lock< std::mutex > l( m );
            wait_flag.clear(std::memory_order_release);
            l.unlock();
            cv.notify_one();
        }
        else
        {
            wait_flag.clear(std::memory_order_release);
        }
    }
};

} // namespace redGrapes

