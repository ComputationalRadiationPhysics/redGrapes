#include <boost/test/unit_test.hpp>

#include <iostream>
#include <rmngr/matrix.hpp>

BOOST_AUTO_TEST_SUITE(matrix);

using namespace rmngr;

BOOST_AUTO_TEST_CASE(matrix)
{
    Matrix<int> m(10, 3);

    for(int r = 0; r < 3; ++r)
    {
        for(int c = 0; c < 10; ++c)
        {
            m(r,c) = 10*(r+1) + c;
        }
    }

    for(int r = 0; r < 3; ++r)
    {
        int c = 0;
        for(int a : m.row(r))
        {
            BOOST_CHECK( a == (10*(r+1) + c) );
            ++c;
        }
    }
}

BOOST_AUTO_TEST_SUITE_END();

