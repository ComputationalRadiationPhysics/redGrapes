#include <boost/test/unit_test.hpp>

#include <iostream>
#include <boost/thread.hpp>
#include <rmngr/thread_dispatcher.hpp>

BOOST_AUTO_TEST_SUITE(thread_dispatcher);

BOOST_AUTO_TEST_CASE(thread_dispatcher)
{
    struct Item
    {
        int id;
        void operator() (void)
        {
            std::cout << "Dispatched " << this->id << std::endl;
        }
    };
    rmngr::ThreadDispatcher<Item, 2, boost::thread> dispatcher;
    dispatcher.push({10});
    dispatcher.push({20});
    dispatcher.push({30});
    dispatcher.push({40});
    dispatcher.push({50});
    dispatcher.push({60});
    dispatcher.push({70});
}

BOOST_AUTO_TEST_SUITE_END();

