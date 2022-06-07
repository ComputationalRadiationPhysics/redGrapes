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

#include <redGrapes/task/allocator.hpp>
#include <redGrapes/task/task.hpp>

namespace redGrapes
{

struct EmplacementQueue
{    
    Task * head;
    Task * tail;

    std::mutex m;
    
    EmplacementQueue();
    
    void push(Task * item);
    Task * pop();
};

/*!
 */
struct TaskSpace : std::enable_shared_from_this<TaskSpace>
{
    /* task storage */
    memory::Allocator task_storage;
    std::atomic< unsigned long > task_count;

    /* queue */
    std::mutex emplacement_mutex;
    EmplacementQueue queue;

    unsigned depth;
    Task * parent;

    // debug
    int next_id;

    virtual ~TaskSpace();
    
    // top space
    TaskSpace();

    // sub space
    TaskSpace( Task & parent );

    virtual bool is_serial( Task& a, Task& b );
    virtual bool is_superset( Task& a, Task& b );

    template < typename F >
    Task & emplace_task( F&& f, TaskProperties&& prop )
    {
        // allocate memory
        FunTask<F> * item = task_storage.m_alloc<FunTask<F>>();
        if( ! item )
            throw std::runtime_error("out of memory");

        ++ task_count;

        // construct task in-place
        new (item) FunTask<F> ( std::move(f), std::move(prop) );

        item->space = shared_from_this();
        item->task = item;
        item->next = nullptr;

        if( parent )
            assert( is_superset(*parent, *item) );

        queue.push(item);

        top_scheduler->notify();
        
        return *item;
    }    
    
    /*! take tasks from the emplacement queue and initialize them,
     *  until a task is initialized whose execution could start immediately
     *
     * @return true if ready task found,
     *         false if no tasks available
     */
    bool init_until_ready();

    void try_remove( Task & task );

    bool empty() const;
};

} // namespace redGrapes
