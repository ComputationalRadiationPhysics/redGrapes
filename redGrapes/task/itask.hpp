/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <memory>

namespace redGrapes
{

struct ITask
{
    virtual ~ITask() {}
    virtual void run() {};
};

struct Task;
/*
//! to access full task from a property
struct TaskProperty
{
    virtual ~TaskProperty() {}


};
*/
} // namespace redGrapes

