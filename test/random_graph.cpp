/* Copyright 2023-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "sha256.c"

#include <redGrapes/SchedulerDescription.hpp>
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace std::chrono;

void sleep(std::chrono::microseconds d)
{
    std::this_thread::sleep_for(d);
}

void hash(unsigned task_id, std::array<uint64_t, 8>& val)
{
    val[0] += task_id;

    uint32_t state[8]
        = {0x6a09'e667, 0xbb67'ae85, 0x3c6e'f372, 0xa54f'f53a, 0x510e'527f, 0x9b05'688c, 0x1f83'd9ab, 0x5be0'cd19};

    sha256_process(state, (uint8_t*) &val[0], sizeof(val));
}

std::chrono::microseconds task_duration(2);
unsigned n_resources = 16;
unsigned n_tasks = 128;
unsigned n_threads = 8;
unsigned min_dependencies = 0;
unsigned max_dependencies = 5;
std::mt19937 gen;

std::vector<std::vector<unsigned>> access_pattern;
std::vector<std::array<uint64_t, 8>> expected_hash;

void generate_access_pattern()
{
    std::uniform_int_distribution<unsigned> distrib_n_deps(min_dependencies, max_dependencies);
    std::uniform_int_distribution<unsigned> distrib_resource(0, n_resources - 1);

    access_pattern = std::vector<std::vector<unsigned>>(n_tasks);
    expected_hash = std::vector<std::array<uint64_t, 8>>(n_resources);
    std::vector<unsigned> path_length(n_resources);

    for(unsigned i = 0; i < n_tasks; ++i)
    {
        unsigned n_dependencies = distrib_n_deps(gen);
        for(unsigned j = 0; j < n_dependencies; ++j)
        {
            unsigned max_path_length = 0;

            while(1)
            {
                unsigned resource_id = distrib_resource(gen);
                if(std::find(access_pattern[i].begin(), access_pattern[i].end(), resource_id)
                   == access_pattern[i].end())
                {
                    access_pattern[i].push_back(resource_id);
                    hash(i, expected_hash[resource_id]);

                    if(path_length[resource_id] > max_path_length)
                        max_path_length = path_length[resource_id];

                    break;
                }
            }

            for(unsigned rid : access_pattern[i])
                path_length[rid] = max_path_length + 1;
        }
    }

    unsigned max_path_length = 1;
    for(unsigned pl : path_length)
        if(pl > max_path_length)
            max_path_length = pl;

    std::cout << "max path length = " << max_path_length << std::endl;
}

TEST_CASE("RandomGraph")
{
    spdlog::set_pattern("[thread %t] %^[%l]%$ %v");

    generate_access_pattern();

    auto rg = redGrapes::init(n_threads);
    {
        std::vector<redGrapes::IOResource<std::array<uint64_t, 8>>> resources(n_resources);

        for(unsigned i = 0; i < n_tasks; ++i)
            switch(access_pattern[i].size())
            {
            case 0:
                rg.emplace_task([]() { sleep(task_duration); });
                break;

            case 1:
                rg.emplace_task(
                    [i](auto ra1)
                    {
                        sleep(task_duration);
                        hash(i, *ra1);
                    },
                    resources[access_pattern[i][0]].write());
                break;

            case 2:
                rg.emplace_task(
                    [i](auto ra1, auto ra2)
                    {
                        sleep(task_duration);
                        hash(i, *ra1);
                        hash(i, *ra2);
                    },
                    resources[access_pattern[i][0]].write(),
                    resources[access_pattern[i][1]].write());
                break;

            case 3:
                rg.emplace_task(
                    [i](auto ra1, auto ra2, auto ra3)
                    {
                        sleep(task_duration);
                        hash(i, *ra1);
                        hash(i, *ra2);
                        hash(i, *ra3);
                    },
                    resources[access_pattern[i][0]].write(),
                    resources[access_pattern[i][1]].write(),
                    resources[access_pattern[i][2]].write());
                break;

            case 4:
                rg.emplace_task(
                    [i](auto ra1, auto ra2, auto ra3, auto ra4)
                    {
                        sleep(task_duration);
                        hash(i, *ra1);
                        hash(i, *ra2);
                        hash(i, *ra3);
                        hash(i, *ra4);
                    },
                    resources[access_pattern[i][0]].write(),
                    resources[access_pattern[i][1]].write(),
                    resources[access_pattern[i][2]].write(),
                    resources[access_pattern[i][3]].write());
                break;

            case 5:
                rg.emplace_task(
                    [i](auto ra1, auto ra2, auto ra3, auto ra4, auto ra5)
                    {
                        sleep(task_duration);
                        hash(i, *ra1);
                        hash(i, *ra2);
                        hash(i, *ra3);
                        hash(i, *ra4);
                        hash(i, *ra5);
                    },
                    resources[access_pattern[i][0]].write(),
                    resources[access_pattern[i][1]].write(),
                    resources[access_pattern[i][2]].write(),
                    resources[access_pattern[i][3]].write(),
                    resources[access_pattern[i][4]].write());
                break;
            }

        rg.barrier();
        for(unsigned i = 0; i < n_resources; ++i)
            REQUIRE(*resources[i].getObject() == expected_hash[i]);
    }
}
