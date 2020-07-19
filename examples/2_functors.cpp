/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <iostream>

#include <redGrapes/manager.hpp>

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
    auto scheduler = std::make_shared<
        redGrapes::scheduler::DefaultScheduler<
            redGrapes::Manager<>::TaskID,
            redGrapes::Manager<>::TaskPtr
        >
    >( 2 /* number of threads */ );

    mgr().set_scheduler( scheduler );

    std::cout << "square(2) = " << square(2).get() << std::endl;

    return 0;
}

