/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <rmngr/resource/ioresource.hpp>
#include <rmngr/property/resource.hpp>
#include <rmngr/property/inherit.hpp>
#include <rmngr/manager.hpp>

using Properties = rmngr::TaskProperties<
    rmngr::ResourceProperty
>;

int main()
{
    rmngr::Manager<
        Properties,
        rmngr::ResourceEnqueuePolicy
    > mgr( 4 );

    rmngr::ResourceBase::scope_level_fn() = [&mgr] { return mgr.scope_level(); };

    rmngr::IOResource a; // scope-level=0
    rmngr::IOResource b; // scope-level=0

    mgr.emplace_task(
        [&mgr]
        {
            rmngr::IOResource c; // scope-level=1

            mgr.emplace_task(
                []{},
                Properties::Builder().resources({ c.write() })
            );
        },
        Properties::Builder().resources({ a.read() })
    );
}
