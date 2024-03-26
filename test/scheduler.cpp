/* Copyright 2023-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>

#include <catch2/catch_test_macros.hpp>
#include <spdlog/spdlog.h>

#include <chrono>


using namespace std::chrono;

void test_worker_utilization(unsigned n_workers)
{
    auto rg = redGrapes::init(n_workers);
    spdlog::set_pattern("[thread %t] %^[%l]%$ %v");

    std::atomic<unsigned> count(0);

    for(unsigned i = 0; i < n_workers; ++i)
    {
        rg.emplace_task(
            [&count]
            {
                count++;
                std::this_thread::sleep_for(milliseconds(300));
            });
    }

    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while(std::chrono::steady_clock::now() < end)
        if(count == n_workers)
            break;

    REQUIRE(count == n_workers);
}

/*
 * create a task for each worker and
 * check that every woker is utilized
 */
TEST_CASE("WorkerUtilization")
{
    for(unsigned i = 1; i < std::thread::hardware_concurrency(); i += 5)
        test_worker_utilization(i);

    test_worker_utilization(std::thread::hardware_concurrency());
}
