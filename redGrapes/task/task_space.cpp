/* Copyright 2021-2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/task/task.hpp>
#include <redGrapes/task/task_space.hpp>

namespace redGrapes
{
    EmplacementQueue::EmplacementQueue() : head(nullptr), tail(nullptr)
    {
    }

    void EmplacementQueue::push(Task* item)
    {
        std::lock_guard<std::mutex> lock(m);
        item->next = nullptr;

        if(tail)
            while(!__sync_bool_compare_and_swap(&(tail->next), nullptr, item))
                ;

        tail = item;

        __sync_bool_compare_and_swap(&head, 0, item);

        SPDLOG_TRACE("push: head = {}, tail = {}", (void*) head, (void*) tail);
    }

    Task* EmplacementQueue::pop()
    {
        std::lock_guard<std::mutex> lock(m);
        while(Task* t = head)
            if(__sync_bool_compare_and_swap(&head, t, t->next))
            {
                SPDLOG_TRACE("queue pop: item={}, new head = {}", (void*) t, (void*) t->next);
                return t;
            }

        SPDLOG_TRACE("pop: head = {}, tail = {}", (void*) head, (void*) tail);
        return nullptr;
    }

    TaskSpace::~TaskSpace()
    {
    }

    TaskSpace::TaskSpace() : active_chunk(0x10000000), parent(nullptr), depth(0), next_id(0)
    {
    }

    // sub space
    TaskSpace::TaskSpace(Task& parent) : active_chunk(0x10000), parent(&parent), depth(parent.space->depth + 1)
    {
    }

    bool TaskSpace::is_serial(Task& a, Task& b)
    {
        return ResourceUser::is_serial(a, b);
    }

    bool TaskSpace::is_superset(Task& a, Task& b)
    {
        return ResourceUser::is_superset(a, b);
    }

    /*! take tasks from the emplacement queue and initialize them,
     *  until a task is initialized whose execution could start immediately
     *
     * @return true if ready task found,
     *         false if no tasks available
     */
    bool TaskSpace::init_until_ready()
    {
        std::lock_guard<std::mutex> lock(emplacement_mutex);

        while(auto task = queue.pop())
        {
            //spdlog::info("redGrapes::TaskSpace:: init task {}", task->task_id);

            if( task->task_id != next_id )
                throw std::runtime_error("invalid next id!");

            next_id++;

            task->alive = 1;
            task->pre_event.up();
            task->init_graph();
            if(task->get_pre_event().notify())
                return true;
        }

        return false;
    }

    void TaskSpace::try_remove(Task& task)
    {
        if(task.post_event.is_reached() && task.result_get_event.is_reached() && (!task.children || task.children->empty()))
        {
            if( __sync_bool_compare_and_swap(&task.alive, 1, 0) )
            {
                task.delete_from_resources();
                task.~Task();
                active_chunk.m_free((void*) &task);

                top_scheduler->notify();
            }
        }

        // todo multiple chunks!
    }

    bool TaskSpace::empty() const
    {
        // todo multiple chunks!
        return active_chunk.empty();
    }

} // namespace redGrapes
