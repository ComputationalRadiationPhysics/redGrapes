
#include <redGrapes/manager.hpp>
#include <redGrapes/resource/fieldresource.hpp>
#include <redGrapes/helpers/mpi/request_pool.hpp>

namespace rg = redGrapes;
/*
using TaskProperties =
    rg::TaskProperties<
        rg::ResourceProperty
    >;
*/
struct MPIConfig
{
    int world_rank;
    int world_size;
};

int main()
{
    rg::Manager< rg::TaskProperties<rg::ResourceProperty>, rg::ResourceEnqueuePolicy > mgr;
    rg::helpers::mpi::RequestPool<decltype(mgr)> mpi_request_pool( mgr );

    // set mpi polling on thread 0
    mgr.getScheduler().schedule[0].set_wait_hook(
        [&mpi_request_pool] {
            mpi_request_pool.poll();
        }
    );

    // initialize MPI
    rg::IOResource< MPIConfig > mpi_config;
    mgr.emplace_task(
        []( auto config ) {
            MPI_Init(nullptr, nullptr);
            MPI_Comm_rank(MPI_COMM_WORLD, &config->world_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &config->world_size);
        },
        mpi_config.write()
    );

    // main loop
    rg::FieldResource< std::array<int, 4> > field[2] = {
        rg::FieldResource<std::array<int, 4>>(new std::array<int, 4>()),
        rg::FieldResource<std::array<int, 4>>(new std::array<int, 4>()),
    };

    int current = 0;

    // initialize
    mgr.emplace_task(
        []( auto buf, auto mpi_config )
        {
            int offset = 3 * mpi_config->world_rank;
            for( size_t i = 0; i < buf->size(); ++i )
                buf[{i}] = offset + i;
        },
        field[current].write(),
        mpi_config.read()
    );

    for(size_t i = 0; i < 8; ++i)
    {
        int next = (current + 1) % 2;

        /*
         * Communication
         */
        // Send
        mgr.emplace_task(
            [i, &mpi_request_pool]( auto field, auto mpi_config )
            {
                auto request = new MPI_Request;
                int dst = ( mpi_config->world_rank + 1 ) % mpi_config->world_size;
                MPI_Isend( &field[{3}], 1, MPI_INT, dst, 0, MPI_COMM_WORLD, request );
                mpi_request_pool.wait( request );
            },
            field[current].at({3}).read(),
            mpi_config.read()
        );
        // Receive
        mgr.emplace_task(
            [i, &mpi_request_pool]( auto field, auto mpi_config )
            {
                auto request = new MPI_Request;
                int src = ( mpi_config->world_rank - 1 ) % mpi_config->world_size;
                MPI_Irecv( &field[{0}], 1, MPI_INT, src, 0, MPI_COMM_WORLD, request );
                mpi_request_pool.wait( request );
            },
            field[current].at({0}).write(),
            mpi_config.read()
        );

        /*
         * Compute iteration
         */
        for( size_t i = 1; i < field[current]->size(); ++i )
            mgr.emplace_task(
                [i]( auto dst, auto src )
                {
                    dst[{i}] = src[{i - 1}];
                },
                field[next].at({i}).write(),
                field[current].at({i-1}).read()
            );

        /*
         * Write Output
         */
        mgr.emplace_task(
            [i]( auto buf, auto mpi_config )
            {
                std::cout << "Step[" << i << "], rank[" << mpi_config->world_rank << "] :: ";
                for( size_t i = 0; i < buf->size(); ++i )
                    std::cout << buf[{i}] << "; ";
                std::cout << std::endl;
            },
            field[current].read(),
            mpi_config.read()
        );

        current = next;
    }

    mgr.emplace_task(
        []( auto m )
        {
            MPI_Finalize();
        },
        mpi_config.write()
    );
}

