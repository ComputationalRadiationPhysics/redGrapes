/* Copyright 2023-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <condition_variable>

#ifndef REDGRAPES_CONDVAR_TIMEOUT
#    define REDGRAPES_CONDVAR_TIMEOUT 0x20'0000
#endif

namespace redGrapes
{

    struct PhantomLock
    {
        inline void lock()
        {
        }

        inline void unlock()
        {
        }
    };

    struct CondVar
    {
        std::atomic<bool> should_wait;
        std::condition_variable cv;
        using CVMutex = std::mutex;
        CVMutex m;

        unsigned timeout;

        CondVar() : CondVar(REDGRAPES_CONDVAR_TIMEOUT)
        {
        }

        CondVar(unsigned timeout) : should_wait(true), timeout(timeout)
        {
        }

        void wait()
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

        bool notify()
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
    };

} // namespace redGrapes
