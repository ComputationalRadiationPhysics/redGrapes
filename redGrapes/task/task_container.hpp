/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <redGrapes/task/task.hpp>
#include <unordered_map>

#include <redGrapes/thread/thread_dispatcher.hpp>

namespace redGrapes
{

template < typename TaskProperties >
class TaskContainer
{
public:
    using TaskID = unsigned int;

    TaskID emplace( Task< TaskProperties > * t )
    {
        std::lock_guard<std::mutex> lock( mutex );
        TaskID id = genTaskID();
        tasks.emplace( id, t );
        return id;
    }

    void erase( TaskID id )
    {
        std::lock_guard<std::mutex> lock( mutex );
        tasks.erase( id );
    }

    TaskProperties & task_properties( TaskID id )
    {
        std::lock_guard<std::mutex> lock( mutex );
        if( tasks.count(id) )
            return tasks[ id ]->properties;
        else
            throw std::runtime_error("TaskContainer: invalid task");
    }

    void task_run( TaskID id )
    {
        std::unique_lock<std::mutex> lock( mutex );
        if( tasks.count(id) )
        {
            auto & task = *tasks[id];
            lock.unlock();
            task();
        }
        else
            throw std::runtime_error("TaskContainer: invalid task");
    }

    void task_hook_before( TaskID id, std::function<void()> const & hook )
    {
        std::lock_guard<std::mutex> lock( mutex );
        if( tasks.count(id) )
            tasks[id]->hook_before( hook );
        else
            throw std::runtime_error("TaskContainer: invalid task");
    }

    void task_hook_after( TaskID id, std::function<void()> const & hook )
    {
        std::lock_guard<std::mutex> lock( mutex );
        if( tasks.count(id) )
            tasks[id]->hook_after( hook );
        else
            throw std::runtime_error("TaskContainer: invalid task");
    }

private:
    std::mutex mutex;
    std::unordered_map<
        TaskID,
        std::unique_ptr< Task< TaskProperties > >
    > tasks;

    static TaskID genTaskID()
    {
        static TaskID next_id = 1;
        if( next_id == 0 )
            throw std::runtime_error("task id overflow");
        return next_id++;
    }
};
    
} // namespace redGrapes
