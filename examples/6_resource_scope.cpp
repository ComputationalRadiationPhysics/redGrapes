/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>

#include <iostream>

namespace rg = redGrapes;

int main()
{
    auto rg = rg::init(1);
    auto a = redGrapes::IOResource<int>(); // scope-level=0

    rg.emplace_task(
          [&]([[maybe_unused]] auto a)
          {
              std::cout << "scope = " << rg.scope_depth() << std::endl;
              auto b = redGrapes::IOResource<int>(); // scope-level=1

              rg.emplace_task(
                    [&](auto b)
                    {
                        *b = 1;
                        std::cout << "scope = " << rg.scope_depth() << std::endl;
                    },
                    b.write())
                  .get();

              std::cout << "scope = " << rg.scope_depth() << std::endl;
          },
          a.read())
        .enable_stack_switching();
}
