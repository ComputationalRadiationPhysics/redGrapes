#include <boost/test/unit_test.hpp>

#include <rmngr/queue.hpp>

BOOST_AUTO_TEST_SUITE(queue);

struct ReadyMarker
{
    void operator() (std::size_t id)
    {
    }
};

BOOST_AUTO_TEST_CASE(push)
{
    ReadyMarker r;
    rmngr::Queue<int, ReadyMarker> queue(r);

    using ID = rmngr::Queue<int, ReadyMarker>::ID;
    std::array<ID, 100> ids;

    for(int i = 0; i < 100; ++i)
        ids[i] = queue.push(new int(2*i));

    for(int i = 0; i < 100; ++i)
        BOOST_ASSERT(queue[ids[i]] == 2*i);
}

BOOST_AUTO_TEST_SUITE_END();

