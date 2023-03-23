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

    std::this_thread::sleep_for(milliseconds(500));

    REQUIRE( count == n_workers );

    rg::finalize();    
}

/*
 * create a task for each worker and
 * check that every woker is utilized
 */
TEST_CASE("WorkerUtilization")
{
    test_worker_utilization(16);
    test_worker_utilization(48);
    test_worker_utilization(63);
    test_worker_utilization(64);
    test_worker_utilization(65);
    test_worker_utilization(120);
    test_worker_utilization(128);
    test_worker_utilization(512);
}

