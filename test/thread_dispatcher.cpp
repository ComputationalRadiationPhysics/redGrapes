
#include <catch/catch.hpp>

#include <mutex>
#include <deque>
#include <functional>
#include <atomic>
#include <redGrapes/thread/thread_dispatcher.hpp>

#include <iostream>

/**
 * Returns n jobs and tests that all jobs are executed
 */
struct TestScheduler
{
    std::vector<bool> worked;

    TestScheduler( int n_threads )
        : worked( n_threads )
    {}

    ~TestScheduler()
    {
    }

    bool empty()
    {
        return true;
    }

    void operator() ()
    {
        worked[redGrapes::thread::id] = true;
    }
};

TEST_CASE("ThreadDispatcher")
{
    for( int n_threads = 1; n_threads < 10; ++n_threads )
    {
        TestScheduler scheduler( n_threads );
        redGrapes::ThreadDispatcher<TestScheduler> dispatcher( scheduler, n_threads );
        dispatcher.finish();
        for( int i = 0; i < n_threads; ++i )
            REQUIRE( scheduler.worked[i] );
    }
}


