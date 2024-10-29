/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#define ENABLE_WORKSTEALING 1

#include <redGrapes/SchedulerDescription.hpp>
#include <redGrapes/dispatch/mpi/mpiWorker.hpp>
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/fieldresource.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/scheduler/mpi_thread_scheduler.hpp>
#include <redGrapes/scheduler/pool_scheduler.hpp>
#include <redGrapes/task/property/label.hpp>

#include <array>
#include <iostream>

/**
 * This example shows how to use MPI with redGrapes.
 *
 * A 1D-array is used, where the first element is
 * synchronized with the last of the left neighbour
 *
 *
 *          Rank 0        |        Rank 1
 *  +---------------------|---------------------+
 *  |  +---+---+---+---+  |  +---+---+---+---+  |
 *  +->| 6 | 1 | 2 | 3 |--|->| 3 | 4 | 5 | 6 |--+
 *     +---+---+---+---+  |  +---+---+---+---+
 *  -->recv|       |send<-|->recv|       |send<--
 *                        |
 *
 * Over this datastructure a very simple iteration
 * is computed: shift all elements one position.
 * For the iteration, double buffering is used.
 */

enum SchedulerTags
{
    SCHED_MPI,
    SCHED_CUDA
};

struct MPIConfig
{
    int world_rank;
    int world_size;
};

struct MPITag
{
};

struct UselessWorkers
{
};

int main()
{
    spdlog::set_pattern("[thread %t] %^[%l]%$ %v");
    spdlog::set_level(spdlog::level::trace);
    using RGTask = redGrapes::Task<>;

    auto rg = redGrapes::init(
        redGrapes::SchedulerDescription(
            std::make_shared<redGrapes::scheduler::PoolScheduler<redGrapes::dispatch::thread::DefaultWorker<RGTask>>>(
                17),
            UselessWorkers{}),
        redGrapes::SchedulerDescription(
            std::make_shared<redGrapes::scheduler::PoolScheduler<redGrapes::dispatch::thread::DefaultWorker<RGTask>>>(
                4),
            redGrapes::DefaultTag{}),
        redGrapes::SchedulerDescription(
            std::make_shared<redGrapes::scheduler::MPIThreadScheduler<RGTask>>(),
            MPITag{}));


    auto& mpiSched = rg.getScheduler<MPITag>();

    auto mpi_request_pool = mpiSched.getRequestPool();

    int prov;

    // initialize MPI
    rg.emplace_task<MPITag>(
        [&prov]()
        {
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &prov);
            assert(prov == MPI_THREAD_FUNNELED);
        });

    // initialize MPI config
    redGrapes::IOResource<MPIConfig> mpi_config;
    rg.emplace_task<MPITag>(
        [](auto config)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &config->world_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &config->world_size);
        },
        mpi_config.write());

    // main loop
    std::array<redGrapes::FieldResource<std::array<int, 4>>, 2> field = {
        redGrapes::FieldResource<std::array<int, 4>>(),
        redGrapes::FieldResource<std::array<int, 4>>(),
    };

    int current = 0;

    // initialize
    rg.emplace_task<MPITag>(
        [](auto buf, auto mpi_config)
        {
            int offset = 3 * mpi_config->world_rank;
            for(size_t i = 0; i < buf->size(); ++i)
                buf[{i}] = offset + i;
        },
        field[current].write(),
        mpi_config.read());

    for(size_t j = 0; j < 4; ++j)
    {
        int next = (current + 1) % 2;

        /*
         * Communication
         */

        // Send
        rg.emplace_task<MPITag>(
              [current, mpi_request_pool](auto field, auto mpi_config)
              {
                  int dst = (mpi_config->world_rank + 1) % mpi_config->world_size;

                  MPI_Request request;
                  MPI_Isend(&field[{3}], sizeof(int), MPI_CHAR, dst, current, MPI_COMM_WORLD, &request);

                  mpi_request_pool->get_status(request);
              },
              field[current].read({3}),
              mpi_config.read())
            .enable_stack_switching();

        // Receive
        rg.emplace_task<MPITag>(
              [current, mpi_request_pool](auto field, auto mpi_config)
              {
                  int src = (mpi_config->world_rank - 1) % mpi_config->world_size;

                  MPI_Request request;
                  MPI_Irecv(&field[{0}], sizeof(int), MPI_CHAR, src, current, MPI_COMM_WORLD, &request);

                  MPI_Status status = mpi_request_pool->get_status(request);

                  int recv_data_count;
                  MPI_Get_count(&status, MPI_CHAR, &recv_data_count);
              },
              field[current].write({0}),
              mpi_config.read())
            .enable_stack_switching();

        /*
         * Compute iteration
         */
        for(size_t i = 1; i < field[current].getObject()->size(); ++i)
            rg.emplace_task(
                [i](auto dst, auto src) { dst[{i}] = src[{i - 1}]; },
                field[next].write({i}),
                field[current].read({i - 1}));

        /*
         * Write Output
         */
        rg.emplace_task(
            [j](auto buf, auto mpi_config)
            {
                std::cout << "Step[" << j << "], rank[" << mpi_config->world_rank << "] :: ";
                for(size_t i = 0; i < buf->size(); ++i)
                    std::cout << buf[{i}] << "; ";
                std::cout << std::endl;
            },
            field[current].read(),
            mpi_config.read());

        current = next;
    }

    rg.emplace_task<MPITag>([]([[maybe_unused]] auto m) { MPI_Finalize(); }, mpi_config.write());
}
