/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <thread>
#include <chrono>
#include <iostream>

#define REDGRAPES_TASK_PROPERTIES redGrapes::LabelProperty

#include <redGrapes/task/property/label.hpp>
#include <redGrapes/redGrapes.hpp>

int main( int, char*[] )
{
    spdlog::set_level(spdlog::level::trace);
    redGrapes::init(4);

    redGrapes::emplace_task(
        []
        {
            std::cout << "f1" << "..." << std::endl;

            int i = 0;
            for( auto t : redGrapes::backtrace() )
            {
                fmt::print("refinement 1 backtrace [{}]: {}\n", i, (redGrapes::TaskProperties const&) t);
            }

            redGrapes::emplace_task(
                []
                {
                    fmt::print("Refinement 1\n");
                    std::this_thread::sleep_for( std::chrono::seconds(1) );
                });

            redGrapes::emplace_task(
                []
                {
                    fmt::print("Refinement 2\n");
                    std::this_thread::sleep_for( std::chrono::seconds(1) );

                    int i = 0;
                    for( auto t : redGrapes::backtrace() )
                    {
                        fmt::print("refinement 2 backtrace [{}]: {}\n", i, (redGrapes::TaskProperties const&) t);
                        i++;
                    }
                },
                redGrapes::TaskProperties::Builder().label("Refinement 2"));
        },
        redGrapes::TaskProperties::Builder().label("Parent Task")
    );

    redGrapes::finalize();

    return 0;
}

