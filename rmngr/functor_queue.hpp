
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
template <typename Queue, typename Updater>
class FunctorQueue
{
    private:
        struct Pusher
        {
            Queue& queue;
            std::mutex & queue_mutex;
          Updater updater;

            template <typename ProtoFunctor, typename DelayedFunctor>
            void operator() (ProtoFunctor const& proto, DelayedFunctor&& delayed)
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                queue.push(proto.clone(std::forward<DelayedFunctor>(delayed)));
                updater();
            }
        }; // struct Pusher

        Pusher pusher;

    public:
  FunctorQueue(Queue& queue, std::mutex & mutex, Updater const& updater)
    : pusher{queue, mutex, updater}
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

  template <typename Queue, typename Updater>
  FunctorQueue<Queue,Updater> make_functor_queue(Queue& queue, std::mutex & mutex, Updater const & up)
{
  return FunctorQueue<Queue,Updater>(queue, mutex, up);
}

}; // namespace rmngr

