
/**
 * @file examples/game_of_life.cpp
 */

#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <rmngr/resource/fieldresource.hpp>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>

using Scheduler =
    rmngr::Scheduler<
        boost::mpl::vector<
            rmngr::ResourceUserPolicy,
          //rmngr::GraphvizWriter< rmngr::Dispatch<rmngr::FIFO> >
            rmngr::DispatchPolicy< rmngr::FIFO >
        >
    >;

Scheduler * scheduler;

static constexpr size_t n_buffers = 4;
struct Position { int x, y; };
static constexpr Position chunk_size{8, 8};
static constexpr Position field_size{chunk_size.x*2, chunk_size.y*2};
using Field = bool[field_size.y+2][field_size.x+2];

// real data
Field * buffers[n_buffers];

// resource representation
rmngr::FieldResource<2> field[n_buffers];

bool
next_state( Field const & neighbours )
{
    int count = neighbours[-1][-1] + neighbours[-1][0] + neighbours[-1][1] +
                neighbours[0][-1] + neighbours[0][1] + neighbours[1][-1] +
                neighbours[1][0] + neighbours[1][1];
    if ( count < 2 || count > 3 )
        return false;
    else if ( count == 3 )
        return true;
    else
        return neighbours[0][0];
}

void
update_cell( Field * dest, Field const * src, Position pos )
{
    Field const & neighbours = *( (Field const *)&( *src )[pos.y][pos.x] );
    ( *dest )[pos.y][pos.x] = next_state( neighbours );
}

void
update_chunk_impl(
    int dst_index,
    int src_index,
    Position pos,
    Position size
)
{
    std::cout << "Buffer " << dst_index << ": calculate chunk " << pos.x << ", " << pos.y << std::endl;
    Field * dest = buffers[dst_index];
    Field const * src = buffers[src_index];
    for ( int x = 0; x < size.x; ++x )
        for ( int y = 0; y < size.y; ++y )
            update_cell( dest, src, Position{pos.x + x, pos.y + y} );

    //std::this_thread::sleep_for(std::chrono::seconds(1));
}

void
update_chunk_prop(
    rmngr::observer_ptr<Scheduler::Schedulable> s,
    int dst_index,
    int src_index,
    Position pos,
    Position size
)
{
    Position end
    {
        pos.x + chunk_size.x - 1,
        pos.y + chunk_size.y - 1,
    };

    scheduler->proto_property< rmngr::ResourceUserPolicy >( s ).access_list =
    {
        field[dst_index].write( {{pos.x, end.x}, {pos.y, end.y}} ),
        field[src_index].read( {{pos.x - 1, end.x + 1}, {pos.y - 1, end.y + 1}} )
    };
}

void
copy_borders_impl(int i)
{
    std::cout << "Buffer " << i << ": copy borders" << std::endl;
    Field * f = buffers[i];
    for ( int x = 0; x < field_size.x + 1; ++x )
    {
        ( *f )[0][x] = ( *f )[field_size.y][x];
        ( *f )[field_size.y + 1][x] = ( *f )[1][x];
    }
    for ( int y = 0; y < field_size.y + 1; ++y )
    {
        ( *f )[y][0] = ( *f )[y][field_size.x];
        ( *f )[y][field_size.x + 1] = ( *f )[y][1];
    }
}

void
copy_borders_prop(
    rmngr::observer_ptr<Scheduler::Schedulable> s,
    int i
)
{
    scheduler->proto_property< rmngr::ResourceUserPolicy >( s ).access_list =
    {
        field[i].write({{0, field_size.x + 1}, {0, 0}}),
        field[i].write({{0, field_size.x + 1}, {field_size.y + 1, field_size.y + 1}}),
        field[i].write({{0, 0}, {0, field_size.y + 1}}),
        field[i].write({{field_size.x + 1, field_size.x + 1}, {0, field_size.y + 1}}),
        field[i].read({{0, field_size.x + 1}, {0, 1}}),
        field[i].read({{0, field_size.x + 1}, {field_size.y, field_size.y + 1}}),
        field[i].read({{0, 1}, {0, field_size.y + 1}}),
        field[i].read({{field_size.x, field_size.x + 1}, {0, field_size.y + 1}}),
    };
}

void
print_buffer_impl( int i )
{
    std::cout << "Print buffer " << i << std::endl;
    Field * const field = buffers[i];

    for ( auto const & row : ( *field ) )
    {
        for ( bool cell : row )
            std::cout << ( cell ? "\x1b[47m" : "\x1b[100m" ) << "  ";
        std::cout << "\x1b[0m" << std::endl;
    }
    std::cout << std::endl;
}

void
print_buffer_prop(
    rmngr::observer_ptr<Scheduler::Schedulable> s,
    int i
)
{
    scheduler->proto_property< rmngr::ResourceUserPolicy >( s ).access_list =
    {
        field[i].read( {{0, field_size.x-1}, {0, field_size.y-1}} )
    };
}

int
main( int, char * [] )
{
    size_t n_threads = std::thread::hardware_concurrency();
    std::cout << "using " << n_threads << " threads." << std::endl;
    scheduler = new Scheduler( n_threads );
    auto queue = scheduler->get_main_queue();

    auto copy_borders =
        queue.make_functor(
            scheduler->make_proto( &copy_borders_impl, &copy_borders_prop )
        );
    auto update_chunk =
        queue.make_functor(
            scheduler->make_proto( &update_chunk_impl, &update_chunk_prop )
        );
    auto print_buffer =
        queue.make_functor(
            scheduler->make_proto( &print_buffer_impl, &print_buffer_prop )
        );

    for ( int i = 0; i < n_buffers; ++i )
        buffers[i] = (Field *)calloc(
                         (field_size.x+2) * (field_size.y+2),
                         sizeof( bool )
                     );

    std::default_random_engine generator;
    std::bernoulli_distribution distribution{0.35};
    for ( int x = 1; x <= field_size.x; ++x )
        for ( int y = 1; y <= field_size.y; ++y )
            ( *buffers[0] )[y][x] = distribution( generator );

    int current_buffer = 0;
    for ( int generation = 0; generation < 10; ++generation )
    {
        copy_borders( current_buffer );
        //print_buffer( current_buffer );
        int next_buffer = ( current_buffer + 1 ) % n_buffers;

        for ( int x = 1; x <= field_size.x; x += chunk_size.x )
        {
            for ( int y = 1; y <= field_size.y; y += chunk_size.y )
            {
                update_chunk(
                    next_buffer,
                    current_buffer,
                    Position{x, y},
                    chunk_size
                );
            }
        }

        current_buffer = next_buffer;
    }

    std::future<void> res = print_buffer( current_buffer );
    res.get();

    for ( int i = 0; i < n_buffers; ++i )
        free( buffers[i] );

    delete scheduler;
    return 0;
}

