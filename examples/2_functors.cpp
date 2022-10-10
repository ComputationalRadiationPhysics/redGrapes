/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <iostream>

//#define REDGRAPES_TASK_PROPERTIES redGrapes::LabelProperty

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/task/property/id.hpp>
#include <redGrapes/task/property/resource.hpp>

auto square (int x)
{
    return redGrapes::emplace_task(
        [x]
        {
            fmt::print("hello\n");
            return x*x;
        }
    ).submit();
}

int main()
{
    spdlog::set_level(spdlog::level::trace);
    spdlog::set_pattern("[thread %t] %^[%l]%$ %v");
    redGrapes::init(1);
    
    fmt::print( "square(2) = {}\n", square(2).get() );

    redGrapes::finalize();    
    return 0;
}

