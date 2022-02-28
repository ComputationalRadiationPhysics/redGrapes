/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <thread>
#include <chrono>
#include <iostream>

#include <redGrapes/task/property/resource.hpp>
#include <redGrapes/task/property/label.hpp>
#include <redGrapes/task/property/inherit.hpp>
#include <redGrapes/redGrapes.hpp>

int main( int, char*[] )
{
    redGrapes::RedGrapes< redGrapes::LabelProperty > rg;
    using TaskProperties = decltype( rg )::TaskProps;

    rg.emplace_task(
        [&rg]
        {
            std::cout << "f1" << "..." << std::endl;
            
            int i = 0;
            for( auto t : rg.backtrace() )
            {
                fmt::print("refinement 1 backtrace [{}]: {}\n", i, (TaskProperties const&)t);
                i++;
            }

            rg.emplace_task(
                []{
                    fmt::print("Refinement 1\n");
                    std::this_thread::sleep_for( std::chrono::seconds(1) );
                });

            rg.emplace_task(
                [&rg]{
                    fmt::print("Refinement 2\n");
                    std::this_thread::sleep_for( std::chrono::seconds(1) );

                    int i = 0;
                    for( auto t : rg.backtrace() )
                    {
                        fmt::print("refinement 2 backtrace [{}]: {}\n", i, (TaskProperties const&)t);
                        i++;
                    }
                },
                TaskProperties::Builder().label("Refinement 2")
            );
        },
        TaskProperties::Builder().label("Parent Task")
    );

    return 0;
}

