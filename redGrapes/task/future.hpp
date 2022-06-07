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

#include <redGrapes/scheduler/event.hpp>
#include <redGrapes/task/task.hpp>

namespace redGrapes
{
    void yield( scheduler::EventPtr event );

    /*!
     * Wrapper for std::future which consumes jobs
     * instead of waiting in get()
     */
    template<typename T>
    struct Future
    {
        Future(Task & task)
            : task( task ),
              taken(false)
        {}

        ~Future()
        {
            if(!taken)
                task.get_result_get_event().notify();
        }

        /*!
         * yields until the task has a valid result
         * and retrieves it.
         *
         * @return the result
         */
        T get(void)
        {
            // wait until result is set
            yield( task.get_result_set_event() );

            // take result
            T result = std::move(*reinterpret_cast<T*>(task->get_result_data()));
            taken = true;
            task.get_result_get_event().notify();
 
            return std::move(result);
        }

        /*! check if the result is already computed
         */
        bool is_ready(void) const
        {
            return task.result_set_event.is_reached();
        }

    private:
        bool taken;
        Task & task;
    }; // struct Future

template<>
struct Future<void>
{
        Future(Task & task)
            : task( task ),
              taken(false)
        {}

        ~Future()
        {
            if(!taken)
                task.get_result_get_event().notify();
        }

        /*!
         * yields until the task has a valid result
         * and retrieves it.
         *
         * @return the result
         */
        void get(void)
        {
            // wait until result is set
            yield( task.get_result_set_event() );

            // take result
            taken = true;
            task.get_result_get_event().notify();
        }

        /*! check if the result is already computed
         */
        bool is_ready(void) const
        {
            return task.result_set_event.is_reached();
        }

    private:
        bool taken;
        Task & task;
};

} // namespace redGrapes
