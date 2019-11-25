/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/working_future.hpp
 */
#pragma once

#include <future>

namespace redGrapes
{

/**
 * Wrapper for std::future which consumes jobs
 * instead of waiting in get()
 *
 * @tparam T delayed type
 * @tparam Worker nullary Callable
 */
template <typename T, typename Worker>
struct WorkingFuture : std::future<T>
{
    WorkingFuture( std::future<T>&& future_, Worker & work_ )
      : std::future<T>(std::move(future_)), work(work_)
    {}

    /**
     * Calls worker until the future has a valid result
     * and retrieves it.
     *
     * @return the result
     */
    T get(void)
    {
        this->work( [this]{ return this->is_ready(); } );
        return this->std::future<T>::get();
    }

    /** check if the result is already computed
     */
    bool is_ready(void) const
    {
        return this->wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

  private:
    Worker & work;
}; // struct WorkingFuture

template <typename T, typename Worker>
WorkingFuture<T, Worker> make_working_future(std::future<T>&& future, Worker & work)
{
    return WorkingFuture<T, Worker>( std::move(future), work );
}

} // namespace redGrapes
