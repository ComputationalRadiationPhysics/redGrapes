/* Copyright 2021 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <functional>
#include <memory>

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/helpers/mpi/request_pool.hpp>
#include <redGrapes/scheduler/fifo.hpp>

namespace redGrapes
{
    namespace helpers
    {
        namespace mpi
        {
            template<typename Manager>
            struct MPIScheduler
            {
                Manager* mgr;
                std::shared_ptr<RequestPool<typename Manager::Task>> request_pool;
                std::shared_ptr<scheduler::FIFO<typename Manager::Task>> fifo;
                typename Manager::Task::Props mpi_props;

                MPIScheduler(Manager& mgr, typename Manager::Task::Props mpi_props)
                    : mgr(&mgr)
                    , request_pool(std::make_shared<RequestPool<typename Manager::Task>>(mgr))
                    , fifo(scheduler::make_fifo_scheduler(mgr))
                    , mpi_props(mpi_props)
                {
                }

                template<typename F>
                TaskResult<typename Manager::Task, MPI_Status> emplace_task(F&& f) const
                {
                    return mgr->emplace_task(
                        [request_pool = this->request_pool, f{std::move(f)}]() -> MPI_Status
                        {
                            MPI_Request request;
                            f(request);
                            return request_pool->get_status(request);
                        },
                        typename Manager::Task::Props::Builder(mpi_props));
                }
            };

            template<typename Manager>
            auto make_mpi_scheduler(Manager& mgr, typename Manager::Task::Props mpi_props)
            {
                return MPIScheduler<Manager>(mgr, std::move(mpi_props));
            }

        } // namespace mpi
    } // namespace helpers
} // namespace redGrapes
