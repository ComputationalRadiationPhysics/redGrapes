/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <thread>
#include <chrono>
#include <iostream>

#include <redGrapes/property/resource.hpp>
#include <redGrapes/property/label.hpp>
#include <redGrapes/property/inherit.hpp>
#include <redGrapes/manager.hpp>

using TaskProperties = redGrapes::TaskProperties<
    redGrapes::LabelProperty,
    redGrapes::ResourceProperty
>;

int main( int, char*[] )
{
    std::ios_base::sync_with_stdio(false);
    redGrapes::Manager<
        TaskProperties,
        redGrapes::ResourceEnqueuePolicy
    > mgr;
    mgr.set_scheduler( redGrapes::scheduler::make_default_scheduler( mgr ) );

    mgr.emplace_task(
        [&mgr]
        {
            std::cout << "f1" << "..." << std::endl;
            
            int i = 0;
            for( auto t : mgr.backtrace() )
            {
                std::cout << "f1 backtrace[" << i << "]: " << std::endl << indent_manip::push << t << indent_manip::pop << std::endl;
                i++;
            }

            mgr.emplace_task(
                []{
                    std::cout << "Refinement 1" << std::endl;
                    std::this_thread::sleep_for( std::chrono::seconds(1) );
                });

            mgr.emplace_task(
                [&mgr]{
                    std::cout << "Refinement 2" << std::endl;
                    std::this_thread::sleep_for( std::chrono::seconds(1) );

                    int i = 0;
                    for( auto t : mgr.backtrace() )
                    {
                        std::cout << "refinement 2 backtrace[" << i << "]: " << std::endl << indent_manip::push << t << indent_manip::pop << std::endl;
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

