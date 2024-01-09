/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/task/property/id.hpp>
#include <redGrapes/task/property/resource.hpp>

#include <iostream>

int square(int x)
{
    return x * x;
}

int main()
{
    spdlog::set_level(spdlog::level::trace);
    redGrapes::init(1);

    fmt::print("square(2) = {}\n", redGrapes::emplace_task(square, 2).get());

    redGrapes::finalize();
    return 0;
}
