#pragma once

#include <functional>
#include <future>
#include <utility>
#include <type_traits>

#include <rmngr/working_future.hpp>

namespace rmngr
{

struct DelayedFunctorInterface
{
    virtual ~DelayedFunctorInterface() {};
    virtual void run (void) = 0;
}; // class DelayedFunctorInterface

template <typename Pusher, typename Functor, typename Worker>
class DelayingFunctor
{
    private:
        template <typename AppliedFunctor, typename Result = void>
        class DelayedFunctor : virtual public DelayedFunctorInterface, public AppliedFunctor
        {
            public:
                DelayedFunctor(AppliedFunctor const& f)
                    : AppliedFunctor(f) {}
                DelayedFunctor(DelayedFunctor&& other)
                    : AppliedFunctor(other), result(std::move(other.result)) {}

                ~DelayedFunctor() {}

                void run (void)
                {
                    set_promise(this->result, *this);
                }

                std::future<Result> get_future(void)
                {
                    return this->result.get_future();
                }

            private:
                std::promise<Result> result;

                template <typename T, typename F>
                static void set_promise (std::promise<T>& p, F& func)
                {
                    p.set_value(func());
                }

                template <typename F>
                static void set_promise (std::promise<void>& p, F& func)
                {
                    func();
                    p.set_value();
                }

        }; // class DelayedFunctor

        template <typename Result, typename AppliedFunctor>
        DelayedFunctor<AppliedFunctor, Result> make_delayed_functor(AppliedFunctor const& f)
        {
            return DelayedFunctor<AppliedFunctor, Result>(f);
        }

        Pusher pusher;
        Functor functor;
        Worker & worker;

    public:
        DelayingFunctor(Pusher const& pusher_, Functor const& functor_, Worker & worker_)
          :  pusher(pusher_), functor(functor_), worker(worker_) {}

        template <typename... Args>
        WorkingFuture<typename std::result_of<Functor(Args...)>::type, Worker>
        operator() (Args&&... args)
        {
            using Result = typename std::result_of<Functor(Args...)>::type;

            auto applied = std::bind(this->functor, std::forward<Args>(args)...);
            auto delayed = make_delayed_functor<Result>(applied);
            auto result = make_working_future(delayed.get_future(), this->worker);
            this->pusher(this->functor, std::move(delayed), std::forward<Args>(args)...);
            return result;
        }
}; // class DelayingFunctor

template <
    typename Pusher,
    typename Functor,
    typename Worker
>
DelayingFunctor<Pusher, Functor, Worker>
make_delaying(
    Pusher const& p,
    Functor const& f,
    Worker& w
)
{
    return DelayingFunctor<Pusher, Functor, Worker>(p, f, w);
}

} // namespace rmngr

