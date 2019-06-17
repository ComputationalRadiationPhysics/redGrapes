
/**
 * @file examples/game_of_life.cpp
 */

#include <array>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <rmngr/resource/fieldresource.hpp>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>

template <typename Graph>
using PrecedenceGraph =
    rmngr::QueuedPrecedenceGraph<
        Graph,
        rmngr::ResourceEnqueuePolicy
    >;

using Scheduler =
    rmngr::Scheduler<
        boost::mpl::vector<
            rmngr::ResourceUserPolicy,
            rmngr::DispatchPolicy< rmngr::FIFO >
        >,
        PrecedenceGraph
    >;

Scheduler * scheduler;

struct Vec2 { int x, y; };
static constexpr size_t size_x = 16;
static constexpr size_t size_y = 16;

enum Cell { DEAD, ALIVE };
struct Buffer : rmngr::FieldResource<2>
{
    Cell (&data)[size_y+2][size_x+2];

    Buffer()
      : data( * ((Cell (*)[size_y+2][size_x+2])malloc(sizeof(Cell)*(size_y+2)*(size_x+2))) )
    {}

    ~Buffer()
    {
        free( &data );
    }
};

Cell
next_state( Cell const neighbours [][size_x+2] )
{
    int count = neighbours[-1][-1] + neighbours[-1][0] + neighbours[-1][1] +
                neighbours[0][-1] + neighbours[0][1] + neighbours[1][-1] +
                neighbours[1][0] + neighbours[1][1];
    if ( count < 2 || count > 3 )
        return DEAD;
    else if ( count == 3 )
        return ALIVE;
    else
        return neighbours[0][0];
}

void
update_cell( Cell dst[][size_x+2], Cell const src[][size_x+2], Vec2 pos )
{
    auto neighbours = (Cell const (*)[size_x+2]) &(src[pos.y][pos.x]);
    dst[pos.y][pos.x] = next_state( neighbours );
}

void
update_chunk_impl(
    Buffer & dst,
    Buffer const & src,
    Vec2 pos,
    Vec2 size
)
{
    for ( int x = 0; x < size.x; ++x )
        for ( int y = 0; y < size.y; ++y )
            update_cell( dst.data, src.data, Vec2{ pos.x + x, pos.y + y } );
}

Scheduler::Properties
update_chunk_prop(
    Buffer & dst,
    Buffer const & src,
    Vec2 pos,
    Vec2 size
)
{
    Vec2 end
    {
        pos.x + size.x - 1,
        pos.y + size.y - 1,
    };

    Scheduler::Properties prop;
    prop.policy< rmngr::ResourceUserPolicy >() += dst.write( {{pos.x, end.x}, {pos.y, end.y}} );
    prop.policy< rmngr::ResourceUserPolicy >() += src.read( {{pos.x - 1, end.x + 1}, {pos.y - 1, end.y + 1}} );

    return prop;
}

void
copy_borders_impl( Buffer & buf )
{
    for ( int x = 0; x < size_x+2; ++x )
    {
        buf.data[0][x] = buf.data[size_y][x];
        buf.data[size_y+1][x] = buf.data[1][x];
    }
    for ( int y = 0; y < size_y+2; ++y )
    {
        buf.data[y][0] = buf.data[y][size_x];
        buf.data[y][size_x+1] = buf.data[y][1];
    }
}

Scheduler::Properties
copy_borders_prop(
    Buffer & buf
)
{
    Scheduler::Properties prop;
    #define ADD_ACCESS prop.policy< rmngr::ResourceUserPolicy >() +=

    ADD_ACCESS buf.write({{0, size_x + 1}, {0, 0}});
    ADD_ACCESS buf.write({{0, size_x + 1}, {size_y + 1, size_y + 1}});
    ADD_ACCESS buf.write({{0, 0}, {0, size_y + 1}});
    ADD_ACCESS buf.write({{size_x + 1, size_x + 1}, {0, size_y + 1}});
    ADD_ACCESS buf.read({{0, size_x + 1}, {0, 1}});
    ADD_ACCESS buf.read({{0, size_x + 1}, {size_y, size_y + 1}});
    ADD_ACCESS buf.read({{0, 1}, {0, size_y + 1}});
    ADD_ACCESS buf.read({{size_x, size_x + 1}, {0, size_y + 1}});

    return prop;
}

void
print_buffer_impl( Buffer const & buf )
{
    for ( auto const & row : buf.data )
    {
        for ( Cell cell : row )
            std::cout << ( ( cell == ALIVE ) ? "\x1b[47m" : "\x1b[100m" ) << "  ";
        std::cout << "\x1b[0m" << std::endl;
    }
    std::cout << std::endl;
}

Scheduler::Properties
print_buffer_prop(
    Buffer const & buf
)
{
    Scheduler::Properties prop;
    prop.policy< rmngr::ResourceUserPolicy >() += buf.read();

    return prop;
}

int
main( int, char * [] )
{
    size_t n_threads = std::thread::hardware_concurrency();
    std::cout << "using " << n_threads << " threads." << std::endl;

    Scheduler * scheduler = new Scheduler( n_threads );
    auto copy_borders = scheduler->make_functor( &copy_borders_impl, &copy_borders_prop );
    auto update_chunk = scheduler->make_functor( &update_chunk_impl, &update_chunk_prop );
    auto print_buffer = scheduler->make_functor( &print_buffer_impl, &print_buffer_prop );

    Vec2 const chunk_size { 8, 8 };
    std::array< Buffer, 4 > buffers;

    int current = 0;

    std::default_random_engine generator;
    std::bernoulli_distribution distribution{0.35};
    for ( int x = 1; x <= size_x; ++x )
        for ( int y = 1; y <= size_y; ++y )
            buffers[current].data[y][x] = distribution( generator ) ? ALIVE : DEAD;

    for ( int generation = 0; generation < 10; ++generation )
    {
        int next = ( current + 1 ) % buffers.size();

        copy_borders( std::ref(buffers[current]) );
        print_buffer( std::ref(buffers[current]) );

        for ( int x = 1; x <= size_x; x += chunk_size.x )
        {
            for ( int y = 1; y <= size_y; y += chunk_size.y )
            {
                update_chunk(
                    std::ref(buffers[next]),
                    std::ref(buffers[current]),
                    Vec2{x, y},
                    chunk_size
                );
            }
        }

        current = next;
    }

    delete scheduler;

    return 0;
}

