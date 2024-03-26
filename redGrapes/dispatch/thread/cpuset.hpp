/* Copyright 2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <pthread.h>
#include <sched.h>
#include <spdlog/spdlog.h>

namespace redGrapes
{
    namespace dispatch
    {
        namespace thread
        {

            inline void pin_cpu(unsigned cpuidx)
            {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(cpuidx % CPU_SETSIZE, &cpuset);

                int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                if(rc != 0)
                    spdlog::error("cannot set thread affinity ({})", rc);
            }

            inline void unpin_cpu()
            {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                for(int j = 0; j < 64; ++j)
                    CPU_SET(j, &cpuset);

                int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                if(rc != 0)
                    spdlog::error("cannot set thread affinity ({})", rc);
            }


        } // namespace thread
    } // namespace dispatch
} // namespace redGrapes
