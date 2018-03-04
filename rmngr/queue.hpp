#pragma once

#include <list>
#include <map>
#include <memory>
#include <algorithm>

namespace rmngr
{

template <typename ID>
class Queue
{
    public:
        virtual ~Queue() {}

        virtual void push(ID a)
        {
            this->queue.insert(this->queue.begin(), a);
        }

        virtual void finish(ID a)
        {
            this->queue.erase(std::find(this->queue.begin(), this->queue.end(), a));
        }

    protected:
        std::list<ID> queue;
};

} // namespace rmngr

