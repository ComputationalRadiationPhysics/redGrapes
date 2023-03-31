#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include <redGrapes/util/chunked_list.hpp>

TEST_CASE("chunked list")
{
    redGrapes::ChunkedList< unsigned > l( 32 );

    // initialy empty
    REQUIRE( l.size() == 0 );
    REQUIRE( l.capacity() == 0 );

    // empty iterator should not iterate
    for( auto & x : l ) {
        REQUIRE( false );
    }

    // size remains 0 while capacity increases
    l.reserve( 4096 );
    REQUIRE( l.size() == 0 );
    REQUIRE( l.capacity() >= 4096 );
    REQUIRE( l.free_capacity() >= 4096 );

    // empty iterator should still not iterate
    for( auto & x : l ) {
        REQUIRE( false );
    }

    // insert elements
    for(unsigned i=0; i < 4096; ++i)
        l.push(i);

    // reversed iterator
    for(unsigned j=0; j < 4096; ++j)
    {
        unsigned i = j;
        for(auto it=l.iter_from(i); it.first != it.second; --it.first)
        {
            REQUIRE( *it.first == i );
            i--;
        }
    }

    // remove
    unsigned r1 = 15;
    unsigned r2 = 48;
    unsigned r3 = 49;
    unsigned r4 = 1023;

    l.remove(r1);
    l.remove(r2);
    l.remove(r3);
    l.remove(r4);

    // check that iterator skips removed elements
    for(unsigned j=1; j < 4096; ++j)
    {
        unsigned i = j;
        for(auto it = l.iter_from(j); it.first != it.second; --it.first)
        {
            unsigned x = *it.first;
            if( i == r4 ) i--;
            if( i == r3 ) i--;
            if( i == r2 ) i--;
            if( i == r1 ) i--;

            REQUIRE( x == i );
            i--;
        }
    }
}

