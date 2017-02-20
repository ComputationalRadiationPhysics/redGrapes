#include <boost/test/unit_test.hpp>

#include <iostream>
#include <string>
#include <rmngr/queue.hpp>

BOOST_AUTO_TEST_SUITE(queue);

using namespace rmngr;

BOOST_AUTO_TEST_CASE(q)
{
    struct Check
    {
        static inline bool check(int const& a, int const& b)
        {
            return (a == b);
        }
    };

    struct Label
    {
        static inline std::string getLabel(int const& a)
        {
            return std::to_string(a);
        }
    };

    Queue<int, Check, Label> queue;

    queue.push(10);
    queue.push(20);
    queue.push(20);
    queue.push(4);
    queue.push(20);

    queue.write_dependency_graph(std::cout);
}

BOOST_AUTO_TEST_SUITE_END();

