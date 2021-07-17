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

namespace redGrapes
{

/*!
 * Wrapper for std::future which consumes jobs
 * instead of waiting in get()
 */
template <
    typename T,
    typename Manager
>
struct TaskResult : std::future<T>
{
    TaskResult( std::future<T> && future, Manager & mgr, EventID result_event )
        : std::future<T>(std::move(future)), mgr(mgr), result_event( result_event )
    {}

    /*!
     * yields until the task has a valid result
     * and retrieves it.
     *
     * @return the result
     */
    T get(void)
    {
        mgr.yield( result_event );
        return this->std::future<T>::get();
    }

    /*! check if the result is already computed
     */
    bool is_ready(void) const
    {
        return this->wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

  private:
    Manager & mgr;
    EventID result_event;
}; // struct TaskResult

template <
    typename T,
    typename Manager
>
TaskResult<T, Manager> make_task_result(std::future<T>&& future, Manager & mgr, EventID event )
{
    return TaskResult< T, Manager >( std::move(future), mgr, event );
}

} // namespace redGrapes

