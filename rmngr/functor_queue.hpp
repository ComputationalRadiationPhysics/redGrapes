
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
 */
template <typename Queue>
class FunctorQueue
{
    private:
        struct Pusher
        {
            Queue& queue;
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

    public:
        FunctorQueue(Queue& queue, std::mutex & mutex)
            : pusher{queue, mutex}
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
        DelayingFunctor<Pusher, ProtoFunctor> make_functor(ProtoFunctor const& proto)
        {
            return make_delaying(this->pusher, proto);
        }

}; // class FunctorQueue

template <typename Queue>
FunctorQueue<Queue> make_functor_queue(Queue& queue, std::mutex & mutex)
{
    return FunctorQueue<Queue>(queue, mutex);
}

}; // namespace rmngr

