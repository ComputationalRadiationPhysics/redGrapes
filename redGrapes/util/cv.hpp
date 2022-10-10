
#pragma once

#include <condition_variable>
#include <redGrapes_config.hpp>

#ifndef REDGRAPES_CONDVAR_TIMEOUT
#define REDGRAPES_CONDVAR_TIMEOUT 500000
#endif

namespace redGrapes
{

struct CondVar
{
    std::condition_variable cv;
    std::mutex m;
    std::atomic_flag wait_flag = ATOMIC_FLAG_INIT;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;

    volatile unsigned count;

    CondVar()
        : count(0)
    {
        wait_flag.test_and_set();
    }

    void wait()
    {
        while( wait_flag.test_and_set(std::memory_order_acquire) )
        {
            if( ++count > REDGRAPES_CONDVAR_TIMEOUT )
            {
                busy.clear();

                std::unique_lock< std::mutex > l( m );
                cv.wait( l, [this]{ return !wait_flag.test_and_set(std::memory_order_acquire); } );
                l.unlock();

                count = 0;
                return;
            }
        }
    }

    bool notify()
    {
        std::unique_lock< std::mutex > l( m );
        bool w = wait_flag.test_and_set();
        wait_flag.clear(std::memory_order_release);
        l.unlock();

        if( ! busy.test_and_set() )
        {
            cv.notify_one();
	    return true;
        }
        else
        {
            return w;
        }
    }
};

} // namespace redGrapes

