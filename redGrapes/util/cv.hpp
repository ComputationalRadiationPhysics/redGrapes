
#pragma once

#include <condition_variable>
#include <redGrapes_config.hpp>

#ifndef REDGRAPES_CONDVAR_TIMEOUT
#define REDGRAPES_CONDVAR_TIMEOUT 0x200000
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
    alignas(64) std::atomic<bool> wait_flag;
    std::condition_variable_any cv;

    alignas(64) std::atomic_flag busy;

    CondVar()
        : wait_flag( true )
    {
    }

    inline void wait()
    {
        unsigned count = 0;

        while( wait_flag.load(std::memory_order_acquire) );
        {
            if( ++count > REDGRAPES_CONDVAR_TIMEOUT )
            {
                busy.clear(std::memory_order_release);

                PhantomLock m;
                std::unique_lock< PhantomLock > l( m );
                cv.wait( l );

                busy.test_and_set();
                count = 0;
            }
        }

        wait_flag.store(true);
    }

    inline bool notify()
    {
        bool w = true;
        wait_flag.compare_exchange_strong(w, false, std::memory_order_release);

        if( ! busy.test_and_set(std::memory_order_acquire) )
            cv.notify_one();

        return w;
    }
};

} // namespace redGrapes

