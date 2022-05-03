/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <thread>
#include <chrono>
#include <iostream>

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/scheduler/default_scheduler.hpp>

namespace rg = redGrapes;

int main( int, char*[] )
{
    rg::init_default();
    rg::IOResource< int > a;

    rg::emplace_task(
        []( auto a )
        {
            std::cout << "f1 writes A" << std::endl;
            std::this_thread::sleep_for( std::chrono::seconds(1) );

            std::cout << "f1 now only reads A" << std::endl;
            rg::update_properties(
                rg::TaskProperties::Patch::Builder()
                    .remove_resources({ a.write() })
                    .add_resources({ a.read() })
            );
            std::this_thread::sleep_for( std::chrono::seconds(1) );

            std::cout << "f1 done" << std::endl; 
        },
        a.write()
    );

    rg::emplace_task(
        []( auto a )
        {
            std::cout << "f2 reads A" << std::endl;
            std::cout << "f2 done" << std::endl;
        },
        a.read()
    );

    rg::barrier();
    
    return 0;
}

