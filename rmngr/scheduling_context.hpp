
#pragma once

#include <mutex>
#include <utility>
#include <rmngr/resource_user.hpp>
#include <rmngr/functor.hpp>

#include <rmngr/queue.hpp>
#include <rmngr/thread_dispatcher.hpp>

namespace rmngr
{

template <template <typename, typename, typename...> typename Scheduler = Queue, std::size_t n_threads=1, typename... Args>
class SchedulingContext
{
    private:
        class Schedulable : public ResourceUser, virtual public DelayedFunctorInterface
        {
            public:
                Schedulable(std::vector<std::shared_ptr<ResourceAccess>> const& access_list_, std::string label_= {})
                    : ResourceUser(access_list_), state(pending), label(label_)
                {}

                enum { pending, ready, running, done, } state;
                std::string label;
        }; // class Schedulable

        template <typename DelayedFunctor>
        class SchedulableFunctor : public DelayedFunctor, public Schedulable
        {
            public:
                SchedulableFunctor(DelayedFunctor&& f, std::vector<std::shared_ptr<ResourceAccess>> const& access_list_, std::string label_)
                    : DelayedFunctor(std::forward<DelayedFunctor>(f)), Schedulable(access_list_, label_) {}
        }; // class SchedulableFunctor

        template <typename Functor>
        class ProtoSchedulableFunctor : public ResourceUser, public Functor
        {
            public:
                ProtoSchedulableFunctor(Functor const& f, std::vector<std::shared_ptr<ResourceAccess>> const& resource_list, std::string label_= {})
                    : ResourceUser(resource_list), Functor(f), label(label_) {}

                template <typename DelayedFunctor>
                SchedulableFunctor<DelayedFunctor>* clone(DelayedFunctor&& f) const
                {
                    return new SchedulableFunctor<DelayedFunctor>(std::forward<DelayedFunctor>(f), this->access_list, this->label);
                }

                std::string label;
        }; // class ProtoSchedulableFunctor

        struct Pusher
        {
            SchedulingContext& context;

            template <typename Functor, typename DelayedFunctor>
            void operator() (ProtoSchedulableFunctor<Functor> const& proto, DelayedFunctor&& delayed)
            {
                std::lock_guard<std::mutex> lock(context.queue_mutex);
                context.queue.push(proto.clone(std::forward<DelayedFunctor>(delayed)));
            }
        }; // struct Pusher

        struct ReadyMarker
        {
            using ID = typename Queue<Schedulable, ReadyMarker>::ID;
            SchedulingContext& context;
            void operator() (ID id)
            {
                if(context.queue[id].state == Schedulable::pending)
                {
                    context.queue[id].state = Schedulable::ready;
                    context.dispatcher.push({&context, id});
                }
            }
        }; // struct ReadyMarker

        struct Executor
        {
            using ID = typename Queue<Schedulable, ReadyMarker>::ID;
            SchedulingContext* context;
            ID id;

            void operator() (void)
            {
                context->queue_mutex.lock();
                Schedulable& s = context->queue[id];
                context->queue_mutex.unlock();

                s.state = Schedulable::running;
                s.run();
                s.state = Schedulable::done;

                std::lock_guard<std::mutex> lock(context->queue_mutex);
                context->queue.finish(id);
            };
        }; // struct Executor

        std::mutex queue_mutex;
        Scheduler<Schedulable, ReadyMarker, Args...> scheduler;
        Queue<Schedulable, ReadyMarker>& queue = scheduler;
        ThreadDispatcher<Executor, n_threads> dispatcher;

    public:
        SchedulingContext()
            : scheduler(ReadyMarker({*this})) {}

        template <typename Functor>
        DelayingFunctor<Pusher, ProtoSchedulableFunctor<Functor>> make_functor(Functor const& f, std::vector<std::shared_ptr<ResourceAccess>> const& resource_list= {}, std::string name= {})
        {
            return make_delaying(Pusher({*this}), ProtoSchedulableFunctor<Functor>(f, resource_list, name));
        }

}; // class SchedulingContext

} // namespace rmngr

