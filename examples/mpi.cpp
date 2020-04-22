
#include <redGrapes/manager.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/resource/fieldresource.hpp>
#include <redGrapes/helpers/mpi/request_pool.hpp>

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
    int prov;
    MPI_Init_thread( nullptr, nullptr, MPI_THREAD_MULTIPLE, &prov );
    assert( prov == MPI_THREAD_MULTIPLE );

    rg::Manager<
        rg::TaskProperties< rg::ResourceProperty >,
        rg::ResourceEnqueuePolicy
    > mgr( 4 );
    auto mpi_request_pool = std::make_shared<rg::helpers::mpi::RequestPool<decltype(mgr)>>( mgr );

    // set mpi polling on thread 1
    mgr.getScheduler().schedule[1].set_wait_hook(
        [mpi_request_pool] {
            mpi_request_pool->poll();
        }
    );

    // initialize MPI config
    rg::IOResource< MPIConfig > mpi_config;
    mgr.emplace_task(
        []( auto config ) {
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

    for(size_t i = 0; i < 100; ++i)
    {
        int next = (current + 1) % 2;

        /*
         * Communication
         */
        // Send
        mgr.emplace_task(
            [i, current, mpi_request_pool]( auto field, auto mpi_config )
            {
                int dst = ( mpi_config->world_rank + 1 ) % mpi_config->world_size;

                MPI_Request request;
                MPI_Isend( &field[{3}], sizeof(int), MPI_CHAR, dst, current, MPI_COMM_WORLD, &request );

                mpi_request_pool->wait( request );
            },
            field[current].at({3}).read(),
            mpi_config.read()
        );

        // Receive
        mgr.emplace_task(
            [i, current, &mgr, mpi_request_pool]( auto field, auto mpi_config )
            {
                int src = ( mpi_config->world_rank - 1 ) % mpi_config->world_size;

                MPI_Request request;
                MPI_Irecv( &field[{0}], sizeof(int), MPI_CHAR, src, current, MPI_COMM_WORLD, &request );

                MPI_Status status = mpi_request_pool->wait( request ).get();

                int recv_data_count;
                MPI_Get_count( &status, MPI_CHAR, &recv_data_count );
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

