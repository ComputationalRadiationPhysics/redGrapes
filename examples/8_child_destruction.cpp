/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_OFF

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>

#include <iostream>

int main()
{
    spdlog::set_level(spdlog::level::off);

    auto rg = redGrapes::init(1);
    auto a = redGrapes::IOResource<int>(1);

    rg.emplace_task(
          [&rg]([[maybe_unused]] auto a)
          {
              std::cout << "scope = " << rg.scope_depth() << " a = " << *a << std::endl;
              auto a2 = redGrapes::IOResource(a);
              rg.emplace_task(
                  [&rg](auto a)
                  {
                      *a = 2;
                      std::cout << "scope = " << rg.scope_depth() << " a = " << *a << std::endl;
                  },
                  a2.write());
              rg.emplace_task(
                  [&rg](auto a)
                  {
                      *a = 3;
                      std::cout << "scope = " << rg.scope_depth() << " a = " << *a << std::endl;
                      auto a3 = redGrapes::IOResource(a);
                      rg.emplace_task(
                          [&rg](auto a)
                          {
                              std::cout << "scope = " << rg.scope_depth() << " a = " << *a << std::endl;
                              *a = 4;
                          },
                          a3.write());
                  },
                  a2.write());

              *a = 4;
              std::cout << "scope = " << rg.scope_depth() << " a = " << *a << std::endl;
          },
          a.write())
        .enable_stack_switching();
    rg.emplace_task(
          [&rg]([[maybe_unused]] auto a)
          { std::cout << "scope = " << rg.scope_depth() << " a = " << *a << std::endl; },
          a.read())
        .enable_stack_switching();
}
