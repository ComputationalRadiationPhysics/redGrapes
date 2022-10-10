/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <redGrapes/task/queue.hpp>

namespace redGrapes
{
namespace task
{

    Queue::Queue() : head(nullptr), tail(nullptr)
    {
    }

    void Queue::push(Task* item)
    {
        cq.enqueue(item);
        /*
        item->next = nullptr;
        Task * n = nullptr;
        while(! tail->next.compare_exchange_weak(n, item));
        */

        //        std::lock_guard<std::mutex> lock(m);
        /*
        if( tail )
            tail.load()->next = item;
        tail = item;
        */
        /*
        //std::lock_guard<std::mutex> lock(m);

        item->next = nullptr;

        if(tail)
            while(!__sync_bool_compare_and_swap(&(tail->next), nullptr, item))
                    break;

        tail = item;

        __sync_bool_compare_and_swap(&head, 0, item);

        SPDLOG_TRACE("push: head = {}, tail = {}", (void*) head, (void*) tail);
        */
    }

    Task * Queue::pop()
    {
        Task * t = nullptr;
        if( cq.try_dequeue( t ) )
        {
            /*
            std::lock_guard<std::mutex> lock(m);
            if( t == tail )
                tail = nullptr;
            */
            return t;
        }
        else
            return nullptr;
        
        /*
        std::lock_guard<std::mutex> lock(m);

        while(Task * volatile t = head)
            if(__sync_bool_compare_and_swap(&head, t, t->next))
            {
                SPDLOG_TRACE("queue pop: item={}, new head = {}", (void*) t, (void*) t->next);

                if(t->next == nullptr)
                    tail = nullptr;
                else
                    t->next = nullptr;
                return t;
            }

        SPDLOG_TRACE("pop: head = {}, tail = {}", (void*) head, (void*) tail);
        return nullptr;
        */
    }

}
}

