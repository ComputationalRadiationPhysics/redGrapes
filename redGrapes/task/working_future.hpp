/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/*!
 * @file redGrapes/working_future.hpp
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
struct WorkingFuture : std::future<T>
{
    WorkingFuture( std::future<T> && future, Manager & mgr, typename Manager::EventID result_event )
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
    typename Manager::EventID result_event;
}; // struct WorkingFuture

template <
    typename T,
    typename Manager
>
WorkingFuture<T, Manager> make_working_future(std::future<T>&& future, Manager & mgr, typename Manager::EventID event )
{
    return WorkingFuture< T, Manager >( std::move(future), mgr, event );
}

} // namespace redGrapes

