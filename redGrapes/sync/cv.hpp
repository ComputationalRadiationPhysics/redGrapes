
#pragma once

#include <atomic>
#include <condition_variable>

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

    unsigned timeout;

    CondVar();
    CondVar( unsigned timeout );
        
    void wait();
    bool notify();
};

} // namespace redGrapes

