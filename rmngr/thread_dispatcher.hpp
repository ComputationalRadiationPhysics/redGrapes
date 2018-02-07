#pragma once

#include <atomic>
#include <thread>
#include <array>
#include <boost/lockfree/queue.hpp>

namespace rmngr
{

template <typename Item, int max_threads=1, typename Thread=std::thread>
class ThreadDispatcher : private boost::lockfree::queue<Item>
{
    public:
        ThreadDispatcher()
            : boost::lockfree::queue<Item>(max_threads), running(true)
        {
            for(size_t i = 0; i < max_threads; ++i)
            {
                this->threads[i] = Thread(thread_main, this, i);
                this->working[i] = false;
            }
        }

        ~ThreadDispatcher()
        {
wait:
            for(size_t i = 0; i < max_threads; ++i)
            {
                if(this->working[i]) goto wait;
            }

            running = false;
            for(size_t i = 0; i < max_threads; ++i)
                this->threads[i].join();
        }

        using boost::lockfree::queue<Item>::push;

    private:
        std::atomic_bool running;
        std::array<std::atomic_bool, max_threads> working;
        std::array<Thread, max_threads> threads;

        static void thread_main(ThreadDispatcher* td, size_t id)
        {
            while(td->running)
            {
                td->consume_all([td, id](Item i)
                {
                    td->working[id] = true;
                    i();
                    td->working[id] = false;
                });
            }
        }
}; // class ThreadDispatcher

} // namespace rmngr

