/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/access/io.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/resource/resource.hpp>

#include <chrono>
#include <iostream>
#include <thread>

namespace rg = redGrapes;

int main(int, char*[])
{
    spdlog::set_level(spdlog::level::trace);
    auto rg = rg::init();
    auto a = redGrapes::IOResource<int>();

    // Access demotion is not implemented

    // rg.emplace_task(
    //     [&](auto a)
    //     {
    //         std::cout << "f1 writes A" << std::endl;
    //         std::this_thread::sleep_for(std::chrono::seconds(1));

    //         std::cout << "f1 now only reads A" << std::endl;
    //         rg.update_properties(decltype(rg)::RGTask::TaskProperties::Patch::Builder()
    //                                  .remove_resources({a})
    //                                  .add_resources({rg::newAccess(a,
    //                                  rg::access::IOAccess(rg::access::IOAccess::read))}));
    //         std::this_thread::sleep_for(std::chrono::seconds(1));

    //         std::cout << "f1 done" << std::endl;
    //     },
    //     a.write());

    rg.emplace_task(
        []([[maybe_unused]] auto a)
        {
            std::cout << "f2 reads A" << std::endl;
            std::cout << "f2 done" << std::endl;
        },
        a.read());

    return 0;
}
