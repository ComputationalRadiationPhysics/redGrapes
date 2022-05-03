/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_OFF

#include <iostream>
#include <thread>
#include <chrono>

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/task/property/inherit.hpp>
#include <redGrapes/task/property/resource.hpp>
#include <redGrapes/resource/ioresource.hpp>

int main()
{
    //spdlog::set_level( spdlog::level::trace );

    redGrapes::init_default(1);

    redGrapes::Resource< redGrapes::access::IOAccess > r1;

    auto event_f = redGrapes::emplace_task(
        [] {
            std::cout << "Task 1" << std::endl;
            return redGrapes::create_event();
        },
        redGrapes::TaskProperties::Builder().resources({ r1.make_access(redGrapes::access::IOAccess::write) })
    );

    redGrapes::emplace_task(
        [] {
            std::cout << "Task 2" << std::endl;
        },
        redGrapes::TaskProperties::Builder().resources({ r1.make_access(redGrapes::access::IOAccess::write) })
    );

    auto event = event_f.get();
    std::cout << "Task 1 finished" << std::endl;

    std::this_thread::sleep_for( std::chrono::seconds(1) );

    std::cout << "notify event" << std::endl;
    event->notify();

    return 0;
}

