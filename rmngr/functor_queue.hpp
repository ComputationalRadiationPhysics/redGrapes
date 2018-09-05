
/*
 * @file rmngr/functor_queue.hpp
 */

#pragma once

#include <mutex>
#include <rmngr/functor.hpp>

namespace rmngr
{

/** Decorates a queue with factory-method to create self-pushing functors.
 *
 * @tparam Queue must have push()
 * @tparam Worker functor to call when returned futures wait
 */
template <typename Queue, typename Worker>
class FunctorQueue
{
    private:
        struct Pusher
        {
            Queue & queue;
            std::mutex & queue_mutex;

            template <
                typename ProtoFunctor,
                typename DelayedFunctor,
                typename... Args
            >
            void operator() (
                ProtoFunctor const& proto,
                DelayedFunctor&& delayed,
                Args&&... args
            )
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                queue.push(
                    proto.clone(
                        std::forward<DelayedFunctor>(delayed),
                        std::forward<Args>(args)...
                ));
            }
        }; // struct Pusher

        Pusher pusher;
        Worker & worker;

    public:
        FunctorQueue(Queue& queue, Worker& worker_, std::mutex & mutex)
            : pusher{queue, mutex}, worker(worker_)
        {}

        /**
         * Create an object, which behaves like a function,
         * but enqueues the functor and returns a future.
         *
         * @tparam ProtoFunctor must have a clone() function, which returns the
         *                      element type of the queue
         * @param proto object to be cloned on every push
         * @return callable object
         */
        template <typename ProtoFunctor>
        DelayingFunctor<Pusher, ProtoFunctor, Worker> make_functor(ProtoFunctor const& proto)
        {
            return make_delaying(this->pusher, proto, this->worker);
        }

}; // class FunctorQueue

template <
    typename Queue,
    typename Worker
>
FunctorQueue<Queue, Worker>
make_functor_queue(
    Queue& queue,
    Worker& worker,
    std::mutex & mutex
)
{
    return FunctorQueue<Queue, Worker>(queue, worker, mutex);
}

}; // namespace rmngr

