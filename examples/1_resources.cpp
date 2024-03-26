/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/redGrapes.hpp>

#include <iostream>

int main(int, char*[])
{
    auto rg = redGrapes::init(1);
    using TTask = decltype(rg)::RGTask;

    auto a = rg.createFieldResource<std::vector<int>>();
    auto b = rg.createIOResource<int>();
    auto c = rg.createIOResource<int>();

    redGrapes::ResourceUser<TTask> user1(
        {a.read(), // complete resource
         a.write().area({0}, {10}), // write only indices 0 to 10
         b.write()});

    redGrapes::ResourceUser<TTask> user2({b.read()});

    redGrapes::ResourceUser<TTask> user3({b.read(), c.write()});

    std::cout << "is_serial(user1,user1) = " << redGrapes::ResourceUser<TTask>::is_serial(user1, user1) << std::endl;
    std::cout << "is_serial(user1,user2) = " << redGrapes::ResourceUser<TTask>::is_serial(user1, user2) << std::endl;
    std::cout << "is_serial(user1,user3) = " << redGrapes::ResourceUser<TTask>::is_serial(user1, user3) << std::endl;
    std::cout << "is_serial(user2,user3) = " << redGrapes::ResourceUser<TTask>::is_serial(user2, user3) << std::endl;

    return 0;
}
