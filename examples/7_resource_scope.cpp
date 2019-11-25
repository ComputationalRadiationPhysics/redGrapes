/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/property/resource.hpp>
#include <redGrapes/property/inherit.hpp>
#include <redGrapes/manager.hpp>

using Properties = redGrapes::TaskProperties<
    redGrapes::ResourceProperty
>;

int main()
{
    redGrapes::Manager<
        Properties,
        redGrapes::ResourceEnqueuePolicy
    > mgr( 4 );

    redGrapes::IOResource a; // scope-level=0
    redGrapes::IOResource b; // scope-level=0

    mgr.emplace_task(
        [&mgr]
        {
            std::cout << "scope = " << redGrapes::thread::scope_level << std::endl;
            redGrapes::IOResource c; // scope-level=1

            mgr.emplace_task(
                []{},
                Properties::Builder().resources({ c.write() })
            );
        },
        Properties::Builder().resources({ a.read() })
    );
}
