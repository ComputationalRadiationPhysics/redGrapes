#include <iostream>
#include <catch/catch.hpp>
#include <redGrapes/util/chunked_list.hpp>

TEST_CASE("chunked list")
{
    redGrapes::ChunkedList< char, 32 > l;

    for(unsigned i=0; i < 4096; ++i)
        l.push((char*)i);

    for(unsigned j=1; j < 4096; ++j)
    {
        unsigned long i = j;
        for(auto it=l.iter_from(j); it.first != it.second; ++it.first)
        {
            REQUIRE( (unsigned long)*it.first == i );
            i--;
        }
    }

    unsigned r1 = 15;
    unsigned r2 = 48;
    unsigned r3 = 49;
    unsigned r4 = 1023;

    l.remove(r1);
    l.remove(r2);
    l.remove(r3);
    l.remove(r4);

    for(unsigned j=1; j < 4096; ++j)
    {
        unsigned long i = j;
        for(auto it = l.iter_from(j); it.first != it.second; ++it.first)
        {
            unsigned long x = (unsigned long) *it.first;
            if( i == r4 ) i--;
            if( i == r3 ) i--;
            if( i == r2 ) i--;
            if( i == r1 ) i--;

            REQUIRE( x == i );
            i--;
        }
    }
}

