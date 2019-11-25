/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <iostream>
#include <thread>
#include <chrono>

#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/property/resource.hpp>
#include <redGrapes/property/inherit.hpp>
#include <redGrapes/manager.hpp>

using Properties = redGrapes::TaskProperties<
    redGrapes::ResourceProperty
>;

int main(void)
{
    redGrapes::Manager<
        Properties,
        redGrapes::ResourceEnqueuePolicy
    > mgr( 4 );

    redGrapes::IOResource a;
    redGrapes::IOResource b;

    auto read_a = mgr.make_functor(
        [a]
        {
            std::cout << "Read from A" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        },
        [a]
        {
            return Properties::Builder().resources({ a.read() });
        }
    );

    auto write_a = mgr.make_functor(
        [a]
        {
            std::cout << "Write to A" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        },
        [a]
        {
            return Properties::Builder().resources({ a.write() });
        }
    );

    auto write_b = mgr.make_functor(
        [b]
        {
            std::cout << "Write to B" << std::endl;
        },
        [b]
        {
            return Properties::Builder().resources({ b.write() });
        }
    );

    auto read_ab = mgr.make_functor(
        [a,b]
        {
            std::cout << "Read from A & B" << std::endl;
        },
        [a,b]
        {
            return Properties::Builder().resources({ a.read(), b.read() });
        }
    );

    for(int i = 0; i < 1; ++i)
    {
        write_a();
        write_a();
        read_a();
        read_a();
        write_b();
        read_ab();
    }

    return 0;
}

