
#pragma once

#include <condition_variable>
#include <redGrapes_config.hpp>

#ifndef REDGRAPES_CONDVAR_TIMEOUT
#define REDGRAPES_CONDVAR_TIMEOUT 0x2000
#endif

namespace redGrapes
{

struct PhantomLock
{
    inline void lock() {}
    inline void unlock() {}
};

struct CondVar
{
    std::atomic<bool> should_wait;
    std::condition_variable_any cv;

    std::atomic_flag busy;

    std::mutex m;

    CondVar()
        : should_wait( true )
    {
    }

    inline void wait()
    {
        unsigned count = 0;

        while( should_wait.load(std::memory_order_acquire) )
        {
#if 1
            if( ++count > REDGRAPES_CONDVAR_TIMEOUT )
            {
                //busy.clear(std::memory_order_release);

                if( should_wait )
                {
                    
                    std::unique_lock< std::mutex > l( m );

                    if( should_wait )
                        cv.wait( l );
                }
            }
#endif
        }

        //        busy.test_and_set();
        count = 0;
        should_wait.store(true);
    }

    inline bool notify()
    {
        bool w = true;
        should_wait.compare_exchange_strong(w, false, std::memory_order_release);

        //if( ! busy.test_and_set(std::memory_order_acquire) )
        {
            std::unique_lock< std::mutex > l( m );
            cv.notify_all();
        }

        return w;
    }
};

} // namespace redGrapes

