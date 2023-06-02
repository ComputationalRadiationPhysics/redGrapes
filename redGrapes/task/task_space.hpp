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

#ifndef REDGRAPES_TASK_ALLOCATOR_CHUNKSIZE
#define REDGRAPES_TASK_ALLOCATOR_CHUNKSIZE 0x800000
#endif

namespace redGrapes
{

/*!
 */
struct TaskSpace : std::enable_shared_from_this<TaskSpace>
{
    /* task storage */
    memory::ChunkAllocator task_storage;
    std::atomic< unsigned long > task_count;

    /* queue */
    task::Queue emplacement_queue;

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

    /* Construct a new task in this task space
     *
     * Note: for each task space there is at most one thread calling
     * this function
     *
     * @param f callable functor to execute in this task
     * @param prop Task-Properties (including resource-
     *             access descriptions)
     *
     * @return: reference to new task
     */
    template < typename F >
    FunTask<F> * alloc_task( )
    {
        // allocate memory
        FunTask<F> * task = task_storage.allocate<FunTask<F>>();
        if( ! task )
            throw std::runtime_error("out of memory");

        return task;
    }

    void free_task( Task * task );
    void submit( Task * task );

    /*! take one task from the emplacement queue, initialize it
     *  and if its execution could start immediately, return it.
     *
     * @return pointer to the new task if it is ready
     */
    bool init_dependencies();
    bool init_dependencies( Task* & t, bool claimed = true );

    bool empty() const;
};

} // namespace redGrapes
