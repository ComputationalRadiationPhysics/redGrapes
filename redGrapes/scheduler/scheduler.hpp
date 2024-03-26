/* Copyright 2020-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <spdlog/spdlog.h>

namespace redGrapes
{
    namespace scheduler
    {

        using WakerId = int16_t;

        /*! Scheduler Interface
         */
        template<typename TTask>
        struct IScheduler
        {
            virtual ~IScheduler()
            {
            }

            /*! whats the task dependency type for the edge a -> b (task a precedes task b)
             * @return true if task b depends on the pre event of task a, false if task b depends on the post event of
             * task b.
             */
            virtual bool task_dependency_type(TTask const& a, TTask const& b)
            {
                return false;
            }

            virtual void idle()
            {
            }

            //! add task to the set of to-initialize tasks
            virtual void emplace_task(TTask& task)
            {
            }

            //! add task to ready set
            virtual void activate_task(TTask& task)
            {
            }

            virtual void wake_all()
            {
            }

            virtual bool wake(WakerId id = 0)
            {
                return false;
            }

            virtual unsigned getNextWorkerID()
            {
                return 0;
            }

            // initialize the execution context pointed to by the scheduler
            virtual void init()
            {
            }

            // start the execution context pointed to by the scheduler
            virtual void startExecution()
            {
            }

            // stop the execution context pointed to by the scheduler
            virtual void stopExecution()
            {
            }
        };

    } // namespace scheduler

} // namespace redGrapes
