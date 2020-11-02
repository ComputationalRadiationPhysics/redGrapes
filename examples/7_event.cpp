/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <iostream>
#include <thread>
#include <chrono>

#include <redGrapes/manager.hpp>
#include <redGrapes/property/inherit.hpp>
#include <redGrapes/property/resource.hpp>
#include <redGrapes/resource/ioresource.hpp>

using TaskProperties = redGrapes::TaskProperties< redGrapes::ResourceProperty >;

int main()
{
    redGrapes::Manager< TaskProperties, redGrapes::ResourceEnqueuePolicy > mgr;
    mgr.set_scheduler( redGrapes::scheduler::make_default_scheduler( mgr ) );

    redGrapes::Resource< redGrapes::access::IOAccess > r1;

    auto event_f = mgr.emplace_task(
        [&mgr] {
            std::cout << "Task 1" << std::endl;
            return *mgr.create_event();
        },
        TaskProperties::Builder().resources({ r1.make_access(redGrapes::access::IOAccess::write) })
    );

    mgr.emplace_task(
        [] {
            std::cout << "Task 2" << std::endl;
        },
        TaskProperties::Builder().resources({ r1.make_access(redGrapes::access::IOAccess::write) })
    );

    auto event = event_f.get();
    std::cout << "Task 1 finished" << std::endl;
    std::this_thread::sleep_for( std::chrono::seconds(1) );
    std::cout << "reach event" << std::endl;
    mgr.reach_event( event );

    return 0;
}

