/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_OFF

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/task/property/inherit.hpp>
#include <redGrapes/task/property/resource.hpp>

#include <chrono>
#include <iostream>
#include <thread>

int main()
{
    spdlog::set_level(spdlog::level::trace);
    spdlog::set_pattern("[thread %t] %^[%l]%$ %v");

    redGrapes::init(1);

    redGrapes::Resource<redGrapes::access::IOAccess> r1;

    auto event_f = redGrapes::emplace_task(
                       []
                       {
                           std::cout << "Task 1" << std::endl;
                           return redGrapes::create_event();
                       })
                       .resources({r1.make_access(redGrapes::access::IOAccess::write)})
                       .submit();

    redGrapes::emplace_task([] { std::cout << "Task 2" << std::endl; })
        .resources({r1.make_access(redGrapes::access::IOAccess::write)});

    auto event = event_f.get();
    std::cout << "Task 1 finished" << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "notify event" << std::endl;
    event->notify();

    redGrapes::finalize();

    return 0;
}
