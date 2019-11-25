
#include <catch/catch.hpp>

#include <mutex>
#include <deque>
#include <functional>
#include <atomic>
#include <redGrapes/thread_dispatcher.hpp>

#include <iostream>

/**
 * Returns n jobs and tests that all jobs are executed
 */
struct TestSelector
{
    std::mutex mutex;
    std::atomic_int n;
    std::atomic_int x;

    TestSelector( int n_ )
      : n(n_), x(n_) {}

    ~TestSelector()
    {
        REQUIRE( this->empty() );
    }

    bool empty()
    {
        return (x == 0);
    }

    template <typename Pred>
    auto getJob( Pred const& )
    {
        std::lock_guard<std::mutex> lock( this->mutex );
        if( n > 0 )
        {
            --n;
            return std::function<void()>([&]() { --x; });
	}
	else
	    return std::function<void()>([]() {});
    }
};

TEST_CASE("ThreadDispatcher")
{
    for( int n_threads = 0; n_threads < 10; ++n_threads )
    {
        for( int n_jobs = 0; n_jobs < 50; ++n_jobs )
	{
   	    TestSelector selector( n_jobs );
	    redGrapes::ThreadDispatcher<TestSelector> dispatcher( selector, n_threads );
	}
    }
}


