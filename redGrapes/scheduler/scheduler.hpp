/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <spdlog/spdlog.h>

#include <optional>

namespace redGrapes
{

    struct Task;

    namespace dispatch
    {
        namespace thread
        {
            struct Worker;
        } // namespace thread
    } // namespace dispatch

    namespace scheduler
    {

        using WakerId = int16_t;

        /*! Scheduler Interface
         */
        struct IScheduler
        {
            virtual ~IScheduler()
            {
            }

            /*! whats the task dependency type for the edge a -> b (task a precedes task b)
             * @return true if task b depends on the pre event of task a, false if task b depends on the post event of
             * task b.
             */
            virtual bool task_dependency_type(Task const& a, Task const& b)
            {
                return false;
            }

            virtual void idle()
            {
            }

            //! add task to the set of to-initialize tasks
            virtual void emplace_task(Task& task)
            {
            }

            //! add task to ready set
            virtual void activate_task(Task& task)
            {
            }

            //! give worker work if available
            virtual Task* steal_task(dispatch::thread::Worker& worker)
            {
                return nullptr;
            }

            virtual void wake_all()
            {
            }

            virtual bool wake(WakerId id = 0)
            {
                return false;
            }
        };

    } // namespace scheduler

} // namespace redGrapes
