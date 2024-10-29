/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_OFF

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>

#include <chrono>
#include <iostream>
#include <thread>

int main()
{
    spdlog::set_level(spdlog::level::trace);
    spdlog::set_pattern("[thread %t] %^[%l]%$ %v");

    auto rg = redGrapes::init(1);

    auto r1 = rg.createResource<redGrapes::access::IOAccess>();

    auto event_f = rg.emplace_task(
                         [&]
                         {
                             std::cout << "Task 1" << std::endl;
                             return rg.create_event();
                         })
                       .resources({r1.make_access(redGrapes::access::IOAccess::write)})
                       .submit();

    rg.emplace_task([] { std::cout << "Task 2" << std::endl; })
        .resources({r1.make_access(redGrapes::access::IOAccess::write)});

    auto event = event_f.get();
    std::cout << "Task 1 finished" << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "notify event" << std::endl;
    event->notify();


    return 0;
}
