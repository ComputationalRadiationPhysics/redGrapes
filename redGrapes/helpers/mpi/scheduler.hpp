/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <memory>
#include <functional>
#include <redGrapes/manager.hpp>
#include <redGrapes/scheduler/fifo.hpp>
#include <redGrapes/helpers/mpi/request_pool.hpp>

namespace redGrapes
{
namespace helpers
{
namespace mpi
{

template < typename Manager >
struct MPIScheduler
{
    std::shared_ptr< RequestPool< Manager > > request_pool;
    std::shared_ptr< scheduler::FIFO< typename Manager::TaskID, typename Manager::TaskPtr > > fifo;
    std::function<
        WorkingFuture< MPI_Status, Manager > (
            std::function< void (MPI_Request&) >
        )
    > emplace_mpi_task;
};

template < typename Manager, typename TaskProps >
auto make_mpi_scheduler( Manager & mgr, TaskProps mpi_props )
{
    auto request_pool = std::make_shared< RequestPool< Manager > >( mgr );
    auto fifo = scheduler::make_fifo_scheduler( mgr );
    auto emplace_mpi_task =
        [mpi_props, &mgr, request_pool] ( std::function< void (MPI_Request&) > f ) {
            return mgr.emplace_task(
                [f, request_pool] {
                    MPI_Request request;
                    f( request );
                    return request_pool->get_status( request );
                },
                mpi_props
            );
        };

    return MPIScheduler< Manager >{
        request_pool,
        fifo,
        emplace_mpi_task
    };
}

}
}
}

