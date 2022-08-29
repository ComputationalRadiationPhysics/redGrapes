/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <redGrapes/task/task.hpp>

namespace redGrapes
{
namespace task
{

struct Queue
{
    Task * volatile head;
    Task volatile * volatile tail;

    std::mutex m;

    moodycamel::ConcurrentQueue< Task* > cq;

    Queue();

    void push(Task * task);
    Task * pop();
};

}
}
