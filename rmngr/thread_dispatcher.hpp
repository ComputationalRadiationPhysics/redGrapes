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
                this->threads[i] = Thread(thread_main, this);
        }

        ~ThreadDispatcher()
        {
            while(this->running = !this->empty());
            for(size_t i = 0; i < max_threads; ++i)
                this->threads[i].join();
        }

        using boost::lockfree::queue<Item>::push;

    private:
        std::atomic_bool running;
        std::array<Thread, max_threads> threads;

        static void thread_main(ThreadDispatcher* td)
        {
            while(td->running)
                td->consume_all([](Item i)
            {
                i();
            });
        }
}; // class ThreadDispatcher

} // namespace rmngr

