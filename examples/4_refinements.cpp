/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/redGrapes.hpp>
#include <redGrapes/task/property/label.hpp>

#include <chrono>
#include <iostream>
#include <thread>

int main(int, char*[])
{
    spdlog::set_level(spdlog::level::trace);
    spdlog::set_pattern("[thread %t] %^[%l]%$ %v");

    auto rg = redGrapes::init<redGrapes::LabelProperty>(4);

    rg.emplace_task(
          [&rg]
          {
              std::cout << "f1"
                        << "..." << std::endl;

              int i = 0;
              for(auto t : rg.backtrace())
                  fmt::print("refinement 1 backtrace [{}]: {}\n", i++, t.get().label);

              rg.emplace_task(
                  []
                  {
                      fmt::print("Refinement 1\n");
                      std::this_thread::sleep_for(std::chrono::seconds(1));
                  });

              SPDLOG_TRACE("EX: create next task task");

              rg.emplace_task(
                    [&rg]
                    {
                        fmt::print("Refinement 2\n");
                        std::this_thread::sleep_for(std::chrono::seconds(1));

                        int i = 0;
                        for(auto t : rg.backtrace())
                            fmt::print(
                                "refinement 2 backtrace [{}]: {}\n",
                                i++,
                                (decltype(rg)::RGTask::TaskProperties const&) t); // TODO cleaner way to do this
                    })
                  .label("Child Task 2");
          })
        .label("Parent Task")
        .submit();


    return 0;
}
