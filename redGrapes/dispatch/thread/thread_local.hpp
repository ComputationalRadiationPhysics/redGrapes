/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

namespace redGrapes
{
namespace dispatch
{
namespace thread
{

/// length of the task-backtrace
static thread_local unsigned int scope_level;

static thread_local std::function< void () > idle;

} // namespace dispatch
} // namespace thread
} // namespace redGrapes

