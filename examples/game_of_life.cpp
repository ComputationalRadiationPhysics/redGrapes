
/**
 * @file examples/game_of_life.cpp
 */

#include <cstdlib>
#include <iostream>
#include <random>
#include <rmngr/fieldresource.hpp>
#include <rmngr/scheduling_context.hpp>

rmngr::SchedulingContext * context;

static constexpr size_t n_buffers = 4;
struct Position { int x, y; };
static constexpr Position chunk_size{1024, 1024};
static constexpr Position field_size{chunk_size.x*8, chunk_size.y*8};
using Field = bool[field_size.y + 2][field_size.x + 2];

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
    Field * dest,
    Field const * src,
    Position pos,
    Position size )
{
    for ( int x = 0; x < size.x; ++x )
        for ( int y = 0; y < size.y; ++y )
            update_cell( dest, src, Position{pos.x + x, pos.y + y} );
}

int
print_buffer_impl( Field const * field )
{
  /*
    for ( auto const & row : ( *field ) )
    {
        for ( bool cell : row )
            std::cout << ( cell ? "\x1b[47m" : "\x1b[100m" ) << "  ";
        std::cout << "\x1b[0m" << std::endl;
    }
    std::cout << std::endl;

  */
    return 0;
}

void
copy_borders_impl( Field * f )
{
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

int
main( int, char * [] )
{
    context = new rmngr::SchedulingContext( 1 );
    auto queue = context->get_main_queue();

    rmngr::FieldResource<2> field[n_buffers];

    auto copy_borders_proto = context->make_proto( &copy_borders_impl );
    auto update_chunk_proto = context->make_proto( &update_chunk_impl );
    auto print_buffer_proto = context->make_proto( &print_buffer_impl );

    Field * buffers[n_buffers];
    for ( int i = 0; i < n_buffers; ++i )
        buffers[i] = (Field *)calloc(
            ( field_size.x + 2 ) * ( field_size.y + 2 ), sizeof( bool ) );

    std::default_random_engine generator;
    std::discrete_distribution<bool> distribution{65, 35};
    for ( int x = 0; x <= field_size.x; ++x )
        for ( int y = 0; y <= field_size.y; ++y )
            ( *buffers[0] )[y][x] = distribution( generator );

    int current_buffer = 0;

    for ( int generation = 0; generation < 10; ++generation )
    {
        // copy borders
        copy_borders_proto.access_list = {
            field[current_buffer].write(
                {std::array<int, 2>{0, field_size.x + 1},
                 std::array<int, 2>{0, 0}} ),
            field[current_buffer].write(
                {std::array<int, 2>{0, field_size.x + 1},
                 std::array<int, 2>{field_size.y + 1, field_size.y + 1}} ),
            field[current_buffer].write(
                {std::array<int, 2>{0, 0},
                 std::array<int, 2>{0, field_size.y + 1}} ),
            field[current_buffer].write(
                {std::array<int, 2>{field_size.x + 1, field_size.x + 1},
                 std::array<int, 2>{0, field_size.y + 1}} ),

            field[current_buffer].read(
                {std::array<int, 2>{0, field_size.x + 1},
                 std::array<int, 2>{0, 1}} ),
            field[current_buffer].read(
                {std::array<int, 2>{0, field_size.x + 1},
                 std::array<int, 2>{field_size.y, field_size.y + 1}} ),
            field[current_buffer].read(
                {std::array<int, 2>{0, 1},
                 std::array<int, 2>{0, field_size.y + 1}} ),
            field[current_buffer].read(
                {std::array<int, 2>{field_size.x, field_size.x + 1},
                 std::array<int, 2>{0, field_size.y + 1}} ) };

        copy_borders_proto.label = "Borders " + std::to_string( current_buffer );
        auto copy_borders = queue.make_functor( copy_borders_proto );
        copy_borders( buffers[current_buffer] );

        print_buffer_proto.access_list = {
            field[current_buffer].read(
                {std::array<int, 2>{0, field_size.x+1},
                 std::array<int, 2>{0, field_size.y+1}})};
        print_buffer_proto.label = "Print " + std::to_string( current_buffer );
        auto print_buffer = queue.make_functor( print_buffer_proto );
        print_buffer( buffers[current_buffer] );

        int next_buffer = ( current_buffer + 1 ) % n_buffers;

        for ( int x = 1; x <= field_size.x; )
        {
            for ( int y = 1; y <= field_size.y; )
            {
                update_chunk_proto.access_list = {
                    field[next_buffer].write(
                        {std::array<int, 2>{x, x + chunk_size.x - 1},
                         std::array<int, 2>{y, y + chunk_size.y - 1}} ),
                    field[current_buffer].read(
                        {std::array<int, 2>{x - 1, x + chunk_size.x},
                         std::array<int, 2>{y - 1, y + chunk_size.y}} )};
                update_chunk_proto.label = "Chunk " + std::to_string( x ) +
                                           "," + std::to_string( y ) + " (" +
                                           std::to_string( next_buffer ) + ")";

                auto update_chunk = queue.make_functor( update_chunk_proto );
                update_chunk(
                    buffers[next_buffer],
                    buffers[current_buffer],
                    Position{x, y},
                    chunk_size );

                y += chunk_size.y;
            }
            x += chunk_size.x;
        }

        current_buffer = next_buffer;
    }

    print_buffer_proto.access_list = {
        field[current_buffer].read(
            {std::array<int, 2>{0, field_size.x + 1},
             std::array<int, 2>{0, field_size.y + 1}} )};
    print_buffer_proto.label = "Print " + std::to_string( current_buffer );
    auto print_buffer = queue.make_functor( print_buffer_proto );

    auto res = print_buffer( buffers[current_buffer] );
    res.get();

    for ( int i = 0; i < n_buffers; ++i )
        free( buffers[i] );

    return 0;
}

