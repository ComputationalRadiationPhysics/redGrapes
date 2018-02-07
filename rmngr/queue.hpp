#pragma once

#include <list>
#include <map>
#include <memory>
#include <algorithm>

namespace rmngr
{

template <typename T, typename ReadyMarker>
class Queue
{
    public:
        typedef std::size_t ID;

        Queue(ReadyMarker const& mark_ready_)
            : id_counter(0), mark_ready(mark_ready_)
        {}

        virtual ~Queue()
        {}

        ID push(T* a)
        {
            ID id = ++this->id_counter;
            this->objects[id] = std::unique_ptr<T>(a);
            this->push_(id);
            this->queue.insert(this->queue.begin(), id);
            this->update_ready(id);
            return id;
        }

        void finish(ID id)
        {
            this->finish_(id);
            this->queue.erase(std::find(this->queue.begin(), this->queue.end(), id));
            this->objects.erase(id);
        }

        T& operator[] (ID id)
        {
            return *this->objects[id];
        }

    protected:
        std::list<ID> queue;

        virtual void push_(ID id) {}
        virtual void finish_(ID id) {}

        virtual bool is_ready(ID id)
        {
            return true;
        }

        void update_ready(ID id)
        {
            if(this->is_ready(id))
                this->mark_ready(id);
        }

    private:
        std::map<ID, std::unique_ptr<T>> objects;
        ID id_counter;

        ReadyMarker mark_ready;
};

} // namespace rmngr

