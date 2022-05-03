/* Copyright 2019-2021 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/*!
 * @file redGrapes/task_result.hpp
 */
#pragma once

#include <future>
#include <redGrapes/scheduler/event.hpp>

namespace redGrapes
{
    void yield( scheduler::EventPtr event );

    /*!
     * Wrapper for std::future which consumes jobs
     * instead of waiting in get()
     */
    template<typename Result>
    struct TaskResult : std::future<Result>
    {
        TaskResult(std::future<Result>&& future, scheduler::EventPtr result_event)
            : std::future<Result>(std::move(future))
            , result_event(result_event)
        {
        }

        /*!
         * yields until the task has a valid result
         * and retrieves it.
         *
         * @return the result
         */
        Result get(void)
        {
            yield(result_event);
            return this->std::future<Result>::get();
        }

        /*! check if the result is already computed
         */
        bool is_ready(void) const
        {
            return this->wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        }

    private:
        scheduler::EventPtr result_event;
    }; // struct TaskResult

    template<typename Result>
    TaskResult<Result> make_task_result(std::future<Result>&& future, scheduler::EventPtr event)
    {
        return TaskResult<Result>(std::move(future), event);
    }

} // namespace redGrapes
