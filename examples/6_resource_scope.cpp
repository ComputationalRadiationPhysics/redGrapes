/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/redGrapes.hpp>

#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/task/property/resource.hpp>
#include <redGrapes/task/property/inherit.hpp>

int main()
{
    redGrapes::RedGrapes<> rg;

    redGrapes::IOResource< int > a; // scope-level=0

    rg.emplace_task(
        [&rg]( auto a )
        {
            std::cout << "scope = " << redGrapes::thread::scope_level << std::endl;
            redGrapes::IOResource<int> b; // scope-level=1

            rg.emplace_task(
                []( auto b )
                {
                    *b = 1;
                    std::cout << "scope = " << redGrapes::thread::scope_level << std::endl;
                },
                b.write()
            ).get();

            std::cout << "scope = " << redGrapes::thread::scope_level << std::endl;
        },
        a.read()
    );
}
