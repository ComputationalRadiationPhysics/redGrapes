#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <thread>

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/util/chunked_list.hpp>

TEST_CASE("ChunkedList singlethreaded")
{
    redGrapes::init(1);
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
    {
        l.push(i);
    }

    // reversed iterator
    for(unsigned j=0; j < 4096; ++j)
    {
        unsigned i = j;
        for(auto it=l.begin_from(i); it != l.end(); --it)
        {
            REQUIRE( *it == i );
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

    // check that forward iterator skips removed elements
    unsigned i = 0;
    for(auto it = l.begin(); it != l.end(); --it)
    {
        unsigned x = *it;
        if( i == r4 ) i++;
        if( i == r3 ) i++;
        if( i == r2 ) i++;
        if( i == r1 ) i++;

        REQUIRE( x == i );
        i++;
    }
    
    // check that reversed iterator skips removed elements
    for(unsigned j=1; j < 4096; ++j)
    {
        unsigned i = j;

        // dont start from removed elements
        if( i == r1 || i == r2 || i == r3 || i == r4 ) break;

        // check reversed iterator
        for(auto it = l.begin_from_rev(i); it != l.end(); --it)
        {
            unsigned x = *it;
            if( i == r4 ) i--;
            if( i == r3 ) i--;
            if( i == r2 ) i--;
            if( i == r1 ) i--;

            REQUIRE( x == i );
            i--;
        }
    }

    redGrapes::finalize();
}

TEST_CASE("ChunkedList: push || push")
{
    redGrapes::init(1);
    redGrapes::ChunkedList< int > l( 8 );

    unsigned end = 20000;
    unsigned mid = end / 2;

    auto t1 = std::thread([&l, mid] {
        for( int i = 0; i < mid; ++i )
            l.push(i);
    });

    auto t2 = std::thread([&l, mid, end] {
        for( int i = mid; i < end; ++i )
            l.push(i);
    });

    t1.join();
    t2.join();

    int expect1 = 0;
    int expect2 = mid;

    for( int x : l )
    {
        if( x < mid )
            REQUIRE( x == expect1++ );
        else
            REQUIRE( x == expect2++ );
    }

    REQUIRE( expect1 == mid );
    REQUIRE( expect2 == end );

    redGrapes::finalize();
}

TEST_CASE("ChunkedList: remove || remove")
{
    redGrapes::init(1);
    redGrapes::ChunkedList< int > l( 8 );
    for( int i = 0; i < 20000; ++i )
        l.push(i);

    auto t1 = std::thread([&l] {
        for( int i = 0; i < 10000; ++i )
            l.remove(i*2);
    });

    auto t2 = std::thread([&l] {
        for( int i = 0; i < 10000; ++i )
            l.remove(i*2+1);
    });

    t1.join();
    t2.join();

    REQUIRE( l.size() == 20000 );

    // empty iterator
    for( int x : l )
        REQUIRE(false);

    redGrapes::finalize();
}

TEST_CASE("ChunkedList: push || remove")
{
    redGrapes::init(1);
    redGrapes::ChunkedList< int > l( 8 );

    // start with 10k elements
    for( int i = 0; i < 10000; ++i )
        l.push(i);

    // push another 10k elements
    auto t1 = std::thread([&l] {
        for( int i = 10000; i < 20000; ++i )
            l.push(i);
    });

    // while removing the first 10k elements
    auto t2 = std::thread([&l] {
        for( int i = 0; i < 10000; ++i )
            l.remove(i);
    });

    t1.join();
    t2.join();

    REQUIRE( l.size() == 20000 );

    int expected = 10000;
    for( int x : l )
        REQUIRE( x == expected++ );

    REQUIRE( expected == 20000 );

    redGrapes::finalize();
}

TEST_CASE("ChunkedList: push || iter")
{
    redGrapes::init(1);
    redGrapes::ChunkedList< int > l( 8 );

    // start with 10k elements
    for( int i = 0; i < 10000; ++i )
        l.push(i);

    // push another 10k elements
    auto t1 = std::thread([&l] {
        for( int i = 10000; i < 20000; ++i )
            l.push(i);
    });

    // while iterating over all elements
    auto t2 = std::thread([&l] {
        int expected = 0;
        for( int x : l )
            REQUIRE( x == expected++ );

        REQUIRE( expected >= 10000 );
    });

    t1.join();
    t2.join();
    redGrapes::finalize();
}

TEST_CASE("ChunkedList: remove || iter")
{
    redGrapes::init(1);
    redGrapes::ChunkedList< int > l( 8 );

    // start with 10k elements
    for( int i = 0; i < 10000; ++i )
        l.push(i);

    // remove every other element
    auto t1 = std::thread([&l] {
        for( int i = 0; i < 10000; i += 2 )
            l.remove(i);
    });

    // while iterating over all elements
    auto t2 = std::thread([&l] {
        int expected = 0;
        for( int x : l )
        {
            REQUIRE( (x == expected || x == (expected+1)) );
            expected = x + 1;
        }
    });

    t1.join();
    t2.join();

    redGrapes::finalize();
}

