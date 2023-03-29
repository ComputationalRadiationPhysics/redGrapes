#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cassert>
#include <chrono>
#include <random>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <spdlog/spdlog.h>

namespace rg = redGrapes;
using namespace std::chrono;

void test_worker_utilization( unsigned n_workers )
{
    rg::init(n_workers);
    spdlog::set_pattern("[thread %t] %^[%l]%$ %v");

    std::atomic< unsigned > count(0);

    for( unsigned i=0; i < n_workers; ++i )
    {
        rg::emplace_task(
                         [&count] {
                             count ++;
                             std::this_thread::sleep_for(seconds(1));
                         }
                         );
    }

    std::this_thread::sleep_for(seconds(1));

    REQUIRE( count == n_workers );

    rg::finalize();    
}

/*
 * create a task for each worker and
 * check that every woker is utilized
 */
TEST_CASE("WorkerUtilization")
{
    for( unsigned i = 1; i <= std::thread::hardware_concurrency(); ++i )
        test_worker_utilization(i);
}


