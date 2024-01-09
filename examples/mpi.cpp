#include <redGrapes/dispatch/mpi/request_pool.hpp>
#include <redGrapes/memory/hwloc_alloc.hpp>
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/fieldresource.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/scheduler/default_scheduler.hpp>
#include <redGrapes/scheduler/tag_match.hpp>

namespace rg = redGrapes;

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

struct MPIConfig
{
    int world_rank;
    int world_size;
};

int main()
{
    spdlog::set_pattern("[thread %t] %^[%l]%$ %v");
    spdlog::set_level(spdlog::level::trace);

    /*
    int prov;
    MPI_Init_thread( nullptr, nullptr, MPI_THREAD_MULTIPLE, &prov );
    assert( prov == MPI_THREAD_MULTIPLE );
    */

    MPI_Init(nullptr, nullptr);

    auto default_scheduler = std::make_shared<rg::scheduler::DefaultScheduler>();
    auto mpi_request_pool = std::make_shared<rg::dispatch::mpi::RequestPool>();

    hwloc_obj_t obj = hwloc_get_obj_by_type(redGrapes::SingletonContext::get().hwloc_ctx.topology, HWLOC_OBJ_PU, 1);
    rg::memory::ChunkedBumpAlloc<rg::memory::HwlocAlloc> mpi_alloc(
        rg::memory::HwlocAlloc(redGrapes::SingletonContext::get().hwloc_ctx, obj));
    auto mpi_worker = std::make_shared<rg::dispatch::thread::Worker>(
        mpi_alloc,
        redGrapes::SingletonContext::get().hwloc_ctx,
        obj,
        4);

    // initialize main thread to execute tasks from the mpi-queue and poll
    rg::SingletonContext::get().idle = [mpi_worker, mpi_request_pool]
    {
        mpi_request_pool->poll();

        redGrapes::Task* task;

        if(task = mpi_worker->ready_queue.pop())
            redGrapes::SingletonContext::get().execute_task(*task);

        while(mpi_worker->init_dependencies(task, true))
            if(task)
            {
                redGrapes::SingletonContext::get().execute_task(*task);
                break;
            }
    };

    rg::init(4, rg::scheduler::make_tag_match_scheduler().add({}, default_scheduler).add({SCHED_MPI}, mpi_worker));

    // initialize MPI config
    rg::IOResource<MPIConfig> mpi_config;
    rg::emplace_task(
        [](auto config)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &config->world_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &config->world_size);
        },
        mpi_config.write())
        .scheduling_tags(std::bitset<64>().set(SCHED_MPI));

    // main loop
    rg::FieldResource<std::array<int, 4>> field[2] = {
        rg::FieldResource<std::array<int, 4>>(new std::array<int, 4>()),
        rg::FieldResource<std::array<int, 4>>(new std::array<int, 4>()),
    };

    int current = 0;

    // initialize
    rg::emplace_task(
        [](auto buf, auto mpi_config)
        {
            int offset = 3 * mpi_config->world_rank;
            for(size_t i = 0; i < buf->size(); ++i)
                buf[{i}] = offset + i;
        },
        field[current].write(),
        mpi_config.read());

    for(size_t i = 0; i < 1; ++i)
    {
        int next = (current + 1) % 2;

        /*
         * Communication
         */

        // Send
        rg::emplace_task(
            [i, current, mpi_request_pool](auto field, auto mpi_config)
            {
                int dst = (mpi_config->world_rank + 1) % mpi_config->world_size;

                MPI_Request request;
                MPI_Isend(&field[{3}], sizeof(int), MPI_CHAR, dst, current, MPI_COMM_WORLD, &request);

                mpi_request_pool->get_status(request);
            },
            field[current].at({3}).read(),
            mpi_config.read())
            .scheduling_tags({SCHED_MPI})
            .enable_stack_switching();

        // Receive
        rg::emplace_task(
            [i, current, mpi_request_pool](auto field, auto mpi_config)
            {
                int src = (mpi_config->world_rank - 1) % mpi_config->world_size;

                MPI_Request request;
                MPI_Irecv(&field[{0}], sizeof(int), MPI_CHAR, src, current, MPI_COMM_WORLD, &request);

                MPI_Status status = mpi_request_pool->get_status(request);

                int recv_data_count;
                MPI_Get_count(&status, MPI_CHAR, &recv_data_count);
            },
            field[current].at({0}).write(),
            mpi_config.read())
            .scheduling_tags({SCHED_MPI})
            .enable_stack_switching();

        /*
         * Compute iteration
         */
        for(size_t i = 1; i < field[current]->size(); ++i)
            rg::emplace_task(
                [i](auto dst, auto src) { dst[{i}] = src[{i - 1}]; },
                field[next].at({i}).write(),
                field[current].at({i - 1}).read());

        /*
         * Write Output
         */
        rg::emplace_task(
            [i](auto buf, auto mpi_config)
            {
                std::cout << "Step[" << i << "], rank[" << mpi_config->world_rank << "] :: ";
                for(size_t i = 0; i < buf->size(); ++i)
                    std::cout << buf[{i}] << "; ";
                std::cout << std::endl;
            },
            field[current].read(),
            mpi_config.read());

        current = next;
    }

    rg::emplace_task([](auto m) { MPI_Finalize(); }, mpi_config.write()).scheduling_tags({SCHED_MPI});

    rg::finalize();
}
