/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <iostream>

#include <redGrapes/manager.hpp>
#include <redGrapes/property/id.hpp>
#include <redGrapes/property/resource.hpp>

static auto & mgr()
{
    static redGrapes::Manager<> m;
    return m;
}

auto square (int x)
{
    return mgr().emplace_task(
        [x]
        {
            return x*x;
        }
    );
}

int main()
{
    spdlog::set_level(spdlog::level::debug);

    fmt::print( "square(2) = {}\n", square(2).get() );

    return 0;
}

