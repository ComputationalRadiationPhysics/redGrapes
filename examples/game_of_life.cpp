/* Copyright 2019-2024 Michael Sippel, Sergei Bastrakov, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file examples/game_of_life.cpp
 */

#include "redGrapes/resource/fieldresource.hpp"

#include <redGrapes/redGrapes.hpp>

#include <array>
#include <cstdlib>
#include <iostream>
#include <random>

struct Vec2
{
    int x, y;
};

enum Cell
{
    DEAD,
    ALIVE
};

static constexpr Vec2 size{32, 32};
static constexpr Vec2 chunk_size{4, 4};

Cell next_state(Cell const neighbours[][size.x + 2])
{
    int count = neighbours[-1][-1] + neighbours[-1][0] + neighbours[-1][1] + neighbours[0][-1] + neighbours[0][1]
                + neighbours[1][-1] + neighbours[1][0] + neighbours[1][1];
    if(count < 2 || count > 3)
        return DEAD;
    else if(count == 3)
        return ALIVE;
    else
        return neighbours[0][0];
}

int main(int, char*[])
{
    spdlog::set_level(spdlog::level::trace);
    spdlog::set_pattern("[thread %t] %^[%l]%$ %v");

    auto rg = redGrapes::init(4);

    using Buffer = std::array<std::array<Cell, size.x + 2>, size.y + 2>;

    std::vector<redGrapes::FieldResource<Buffer>> buffers;

    for(size_t i = 0; i < 4; ++i)
        buffers.emplace_back();

    int current = 0;

    // initialization
    rg.emplace_task(
        [](auto buf)
        {
            std::default_random_engine generator;
            std::bernoulli_distribution distribution{0.35};

            for(size_t x = 0; x < size.x + 2; ++x)
                for(size_t y = 0; y < size.y + 2; ++y)
                    buf[{x, y}] = distribution(generator) ? ALIVE : DEAD;
        },
        buffers[current].write());

    for(int generation = 0; generation < 500; ++generation)
    {
        int next = (current + 1) % buffers.size();

        // copy borders
        rg.emplace_task(
            [](auto buf)
            {
                for(size_t x = 0; x < size.x + 2; ++x)
                {
                    buf[{x, 0}] = buf[{x, size.y}];
                    ;
                    buf[{x, size.y + 1}] = buf[{x, 1}];
                }
                for(size_t y = 0; y < size.y + 2; ++y)
                {
                    buf[{0, y}] = buf[{size.x, y}];
                    buf[{size.x + 1, y}] = buf[{1, y}];
                }
            },
            buffers[current].write());

        // print buffer
        rg.emplace_task(
              [](auto buf)
              {
                  for(size_t x = 1; x < size.x; ++x)
                  {
                      for(size_t y = 1; y < size.y; ++y)
                      {
                          std::cout << ((buf[{x, y}] == ALIVE) ? "[47m" : "[100m") << "  ";
                      }
                      std::cout << "[0m" << std::endl;
                  }
                  std::cout << std::endl;
              },
              buffers[current].read())
            .get();

        // calculate next step
        for(size_t x = 1; x <= size.x; x += chunk_size.x)
            for(size_t y = 1; y <= size.y; y += chunk_size.y)
                rg.emplace_task(
                    [x, y](auto dst, auto src)
                    {
                        for(int xi = 0; xi < chunk_size.x; ++xi)
                            for(int yi = 0; yi < chunk_size.y; ++yi)
                                dst[{x + xi, y + yi}]
                                    = next_state((Cell const(*)[size.x + 2]) & (src[{x + xi, y + yi}]));
                    },
                    buffers[next].write({x, y}, {x + chunk_size.x, y + chunk_size.y}),
                    buffers[current].read({x - 1, y - 1}, {x + chunk_size.x + 2, y + chunk_size.y + 2}));

        current = next;
    }

    SPDLOG_DEBUG("END!!!!");

    return 0;
}
