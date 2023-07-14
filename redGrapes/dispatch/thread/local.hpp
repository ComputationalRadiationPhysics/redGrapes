/* Copyright 2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <redGrapes/scheduler/scheduler.hpp>

namespace redGrapes
{
namespace dispatch
{
namespace thread
{

struct WorkerThread;

extern thread_local scheduler::WakerID current_waker_id;
extern thread_local std::shared_ptr< WorkerThread > current_worker;

} // namespace thread
} // namespace dispatch
} // namespace redGrapes

