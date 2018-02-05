#pragma once

#include <list>
#include <map>
#include <memory>

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

        virtual ID push(T* a)
        {
            auto it = this->queue.insert(this->queue.begin(), std::unique_ptr<T>(a));
            ID id = ++this->id_counter;
            this->id_map[id] = it;
            this->update_ready(id);
            return id;
        }

        virtual void finish(ID id)
        {
            auto it = this->id_map[id];
            this->queue.erase(it);
            this->id_map.erase(id);
        }

        T& operator[] (ID id)
        {
            return **this->id_map[id];
        }

    protected:
        std::list<std::unique_ptr<T>> queue;

        virtual bool is_ready(ID id) const
        {
            return true;
        }

        void update_ready(ID id)
        {
            if(this->is_ready(id))
                this->mark_ready(id);
        }

    private:
        std::map<ID, typename std::list<typename std::unique_ptr<T>>::iterator> id_map;
        ID id_counter;

        ReadyMarker mark_ready;
};

} // namespace rmngr

