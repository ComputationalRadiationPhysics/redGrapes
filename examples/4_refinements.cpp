/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <thread>
#include <chrono>
#include <iostream>

#include <rmngr/property/resource.hpp>
#include <rmngr/property/inherit.hpp>
#include <rmngr/manager.hpp>

using Properties = rmngr::TaskProperties<
    rmngr::ResourceProperty
>;

int main( int, char*[] )
{
    rmngr::Manager<
        Properties,
        rmngr::ResourceEnqueuePolicy
    > mgr( 4 );

    auto fun1 = mgr.make_functor(
        [&mgr]
        {
            std::cout << "f1 on thread " << rmngr::thread::id << "..." << std::endl;

            mgr.emplace_task(
                []{
                    std::cout << "Refinement 1 on thread " << rmngr::thread::id << std::endl;
                    std::this_thread::sleep_for( std::chrono::seconds(1) );
                });

            mgr.emplace_task(
                []{
                    std::cout << "Refinement 2 on thread " << rmngr::thread::id << std::endl;
                    std::this_thread::sleep_for( std::chrono::seconds(1) );
                });
        }
    );

    fun1();
    fun1();

    return 0;
}

