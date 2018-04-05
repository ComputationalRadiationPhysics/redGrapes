
/*
 * @file rmngr/functor_queue.hpp
 */

#pragma once

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

            template <typename ProtoFunctor, typename DelayedFunctor>
            void operator() (ProtoFunctor const& proto, DelayedFunctor&& delayed)
            {
                queue.push(proto.clone(std::forward<DelayedFunctor>(delayed)));
            }
        }; // struct Pusher

        Pusher pusher;

    public:
        FunctorQueue(Queue& queue)
            : pusher{queue}
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
FunctorQueue<Queue> make_functor_queue(Queue& queue)
{
    return FunctorQueue<Queue>(queue);
}

}; // namespace rmngr

