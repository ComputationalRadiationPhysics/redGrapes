
/**
 * @file examples/game_of_life.cpp
 */

#include <array>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <rmngr/resource/fieldresource.hpp>
#include <rmngr/resource/ioresource.hpp>
#include <rmngr/property/resource.hpp>
#include <rmngr/property/inherit.hpp>
#include <rmngr/manager.hpp>

using Properties = rmngr::TaskProperties<
    rmngr::ResourceProperty
>;

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

int
main( int, char * [] )
{
    size_t n_threads = std::thread::hardware_concurrency();
    std::cout << "using " << n_threads << " threads." << std::endl;

    auto mgr = new rmngr::Manager<
        Properties,
        rmngr::ResourceEnqueuePolicy
    >( n_threads );

    Vec2 const chunk_size { 4, 4 };
    std::array< Buffer, 4 > buffers;

    int current = 0;

    std::default_random_engine generator;
    std::bernoulli_distribution distribution{0.35};
    for ( int x = 1; x <= size_x; ++x )
        for ( int y = 1; y <= size_y; ++y )
            buffers[current].data[y][x] = distribution( generator ) ? ALIVE : DEAD;

    for ( int generation = 0; generation < 500; ++generation )
    {
        int next = ( current + 1 ) % buffers.size();
        auto & buf = buffers[current];

        mgr->emplace_task(
            [&buf, generation]
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
            },
            Properties::Builder()
                .resources({
                    buf.write({{0, size_x + 1}, {0, 0}}),
                    buf.write({{0, size_x + 1}, {size_y + 1, size_y + 1}}),
                    buf.write({{0, 0}, {0, size_y + 1}}),
                    buf.write({{size_x + 1, size_x + 1}, {0, size_y + 1}}),
                    buf.read({{0, size_x + 1}, {0, 1}}),
                    buf.read({{0, size_x + 1}, {size_y, size_y + 1}}),
                    buf.read({{0, 1}, {0, size_y + 1}}),
                    buf.read({{size_x, size_x + 1}, {0, size_y + 1}})
                })
        );

        // print buffer
        mgr->emplace_task(
            [&buf]
            {
                for ( auto const & row : buf.data )
                {
                    for ( Cell cell : row )
                        std::cout << ( ( cell == ALIVE ) ? "\x1b[47m" : "\x1b[100m" ) << "  ";
                    std::cout << "\x1b[0m" << std::endl;
                }
                std::cout << std::endl;    
            },
            Properties::Builder().resources({ buf.read() })
        );

        // update
        for ( int x = 1; x <= size_x; x += chunk_size.x )
        {
            for ( int y = 1; y <= size_y; y += chunk_size.y )
            {
                auto & dst = buffers[next];
                auto & src = buffers[current];
                mgr->emplace_task(
                    [&dst, &src, x, y, chunk_size]
                    {
                        for ( int xi = 0; xi < chunk_size.x; ++xi )
                            for ( int yi = 0; yi < chunk_size.y; ++yi )
                                update_cell( dst.data, src.data, Vec2{ x + xi, y + yi } );
                    },
                    Properties::Builder()
                        .resources({
                            dst.write( {{x, x + chunk_size.x - 1}, {y, y + chunk_size.y - 1}} ),
                            src.read( {{x - 1, x + chunk_size.x}, {y - 1, y + chunk_size.y}} )
                        })
                );
            }
        }

        current = next;
    }

    delete mgr;

    return 0;
}

