#include <catch/catch.hpp>

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
#include "sha256.c"

namespace rg = redGrapes;
using namespace std::chrono;


void sleep(std::chrono::microseconds d)
{
    auto end = std::chrono::high_resolution_clock::now() + d;
    while(std::chrono::high_resolution_clock::now() < end)
        ;
}

void hash(unsigned task_id,
          std::array<uint64_t, 8> & val)
{
    val[0] += task_id;

    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    sha256_process(state, (uint8_t*)&val[0], sizeof(val));

    for(int i=0; i<8; ++i)
        val[i] = state[i];
}

std::chrono::microseconds task_duration(5);
unsigned n_resources = 5;
unsigned n_tasks = 10000;
unsigned n_threads = 4;
unsigned min_dependencies = 0;
unsigned max_dependencies = 5;
std::mt19937 gen;

std::vector<std::vector<unsigned>> access_pattern;
std::vector<std::array<uint64_t, 8>> expected_hash;

void generate_access_pattern()
{
    std::uniform_int_distribution<unsigned> distrib_n_deps(min_dependencies, max_dependencies);
    std::uniform_int_distribution<unsigned> distrib_resource(0, n_resources - 1);

    expected_hash = std::vector<std::array<uint64_t, 8>>(n_resources);
    std::vector<unsigned> path_length(n_resources);

    for(int i = 0; i < n_tasks; ++i)
    {
        access_pattern.emplace_back();
        unsigned n_dependencies = distrib_n_deps(gen);
        for(int j = 0; j < n_dependencies; ++j)
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

		    if( path_length[resource_id] > max_path_length )
		      max_path_length = path_length[resource_id];

                    break;
                }
            }

	    for( unsigned rid : access_pattern[i] )
	      path_length[rid] = max_path_length + 1;
        }
    }

    unsigned max_path_length = 1;
    for( unsigned pl : path_length )
      if( pl > max_path_length )
	max_path_length = pl;

    std::cout << "max path length = " << max_path_length << std::endl;
}

TEST_CASE("RandomGraph")
{
    generate_access_pattern();
    rg::init(4);
    spdlog::set_pattern("[thread %t] %^[%l]%$ %v");
    
    std::vector<rg::IOResource<std::array<uint64_t, 8>>> resources(n_resources);

    for(int i = 0; i < n_tasks; ++i)
        switch(access_pattern[i].size())
        {
        case 0:
            rg::emplace_task([i]() { sleep(task_duration); });
            break;

        case 1:
            rg::emplace_task(
                [i](auto ra1)
                {
                    sleep(task_duration);
                    hash(i, *ra1);
                },
                resources[access_pattern[i][0]].write());
            break;

        case 2:
            rg::emplace_task(
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
            rg::emplace_task(
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
            rg::emplace_task(
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
            rg::emplace_task(
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

    rg::barrier();
 
    for(int i = 0; i < n_resources; ++i)
        REQUIRE( *resources[i] == expected_hash[i] );
}

