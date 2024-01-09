
#pragma once

#include <atomic>
#include <memory>
#include <mutex>

namespace redGrapes
{

#define SPIN 1

    struct /*alignas(64)*/ SpinLock
    {
#if SPIN
        std::atomic<bool> state;
#else
        std::mutex m;
#endif

        SpinLock()
#if SPIN
            : state(false)
#endif
        {
        }

        inline void lock()
        {
#if SPIN
            while(true)
            {
                bool s = false;
                if(state.compare_exchange_weak(s, true, std::memory_order_acquire))
                    if(s == false)
                        return;

                while(state.load(std::memory_order_relaxed))
                    ;
            }
#else
            m.lock();
#endif
        }

        inline void unlock()
        {
#if SPIN
            state.store(false, std::memory_order_release);
#else
            m.unlock();
#endif
        }
    };

    /*
    struct alignas(64) RWSpinLock
    {
    #if SPIN
        alignas(64) std::atomic<unsigned> reader_count;
        alignas(64) std::atomic<bool> write;
    #else
        std::shared_timed_mutex m;
    #endif

        SpinLock()
    #if SPIN
            : reader_count(0)
            , write(0)
    #endif
        {
        }

        inline void lock_shared()
        {
            reader_count.fetch_add(1);
        }

        inline void unlock_shared()
        {
            reader_count.fetch_sub(1);
        }

        inline void lock()
        {
    #if SPIN
            while( true )
            {
                bool s = false;
                if( write.compare_exchange_weak(s, true, std::memory_order_acquire) )
                    if( s == false )
                        return;

                while( state.load(std::memory_order_relaxed) );
            }
    #else
            m.lock();
    #endif
        }

        inline void unlock()
        {
    #if SPIN
            write.store(false, std::memory_order_release);
    #else
            m.unlock();
    #endif
        }
    };
    */

} // namespace redGrapes
