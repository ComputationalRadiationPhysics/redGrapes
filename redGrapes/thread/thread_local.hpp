/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace redGrapes
{

namespace thread
{

/// identifies the current thread
static thread_local std::size_t id;

/// length of the task-backtrace
static thread_local unsigned int scope_level;

} // namespace thread

} // namespace redGrapes

