/* Copyright 2021-2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <vector>
#include <mutex>

#include <redGrapes/util/allocator.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/task/queue.hpp>
#include <redGrapes/dispatch/thread/cpuset.hpp>

namespace redGrapes
{

/*!
 */
struct TaskSpace : std::enable_shared_from_this<TaskSpace>
{
    std::atomic< unsigned long > task_count;

    unsigned depth;
    Task * parent;

    std::shared_mutex active_child_spaces_mutex;
    std::vector< std::shared_ptr< TaskSpace > > active_child_spaces;

    virtual ~TaskSpace();
    
    // top space
    TaskSpace();

    // sub space
    TaskSpace( Task * parent );

    virtual bool is_serial( Task& a, Task& b );
    virtual bool is_superset( Task& a, Task& b );

    void free_task( Task * task );
    void submit( Task * task );

    bool empty() const;
};

} // namespace redGrapes
