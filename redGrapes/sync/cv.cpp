
#include <redGrapes/sync/cv.hpp>

#include <redGrapes_config.hpp>
#ifndef REDGRAPES_CONDVAR_TIMEOUT
#    define REDGRAPES_CONDVAR_TIMEOUT 0x20'0000
#endif

namespace redGrapes
{
    CondVar::CondVar() : CondVar(REDGRAPES_CONDVAR_TIMEOUT)
    {
    }

    CondVar::CondVar(unsigned timeout) : should_wait(true), timeout(timeout)
    {
    }

    void CondVar::wait()
    {
        unsigned count = 0;
        while(should_wait.load(std::memory_order_acquire))
        {
            if(++count > timeout)
            {
                // TODO: check this opmitization
                // busy.clear(std::memory_order_release);

                if(should_wait.load(std::memory_order_acquire))
                {
                    std::unique_lock<CVMutex> l(m);
                    cv.wait(l, [this] { return !should_wait.load(std::memory_order_acquire); });
                }
            }
        }

        should_wait.store(true);
    }

    bool CondVar::notify()
    {
        bool w = true;
        should_wait.compare_exchange_strong(w, false, std::memory_order_release);

        // TODO: check this optimization
        // if( ! busy.test_and_set(std::memory_order_acquire) )
        {
            std::unique_lock<std::mutex> l(m);
            cv.notify_all();
        }

        return w;
    }

} // namespace redGrapes
