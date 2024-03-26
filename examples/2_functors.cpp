/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/redGrapes.hpp>

int square(int x)
{
    return x * x;
}

int main()
{
    spdlog::set_level(spdlog::level::trace);
    auto rg = redGrapes::init(1);

    fmt::print("square(2) = {}\n", rg.emplace_task(square, 2).get());

    return 0;
}
