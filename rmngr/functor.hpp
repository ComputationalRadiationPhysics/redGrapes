
/**
 * @file rmngr/functor.hpp
 */

#pragma once

#include <functional>
#include <future>
#include <utility>
#include <type_traits>

#include <rmngr/working_future.hpp>

namespace rmngr
{

/**
 * Polymorphic base class for functors created through calling DelayingFunctor.
 */
struct DelayedFunctorInterface
{
    virtual ~DelayedFunctorInterface() {};
    virtual void run (void) = 0;
}; // class DelayedFunctorInterface

/**
 * Wraps a functor to create delayed-functors on call
 * and pass them on to a pusher (which could push them to a queue)
 *
 * @tparam Pusher must be Callable of the form `void (ProtoFunctor const&, DelayedFunctor&, Args&&...)`
 * @tparam Functor must be any Callable
 * @tparam Worker must be a nullary Callable
 */
template <
    typename Pusher,
    typename Functor,
    typename Worker
>
class DelayingFunctor
{
    private:
        template <
            typename AppliedFunctor,
            typename Result = void
        >
        class DelayedFunctor
	    : virtual public DelayedFunctorInterface
	    , public AppliedFunctor
        {
            public:
                DelayedFunctor(AppliedFunctor const& f)
                    : AppliedFunctor(f)
	        {}

                DelayedFunctor(DelayedFunctor&& other)
                    : AppliedFunctor(other)
		    , result(std::move(other.result))
	        {}

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
        /**
         * @param pusher a callable which will receive the functor itself,
         *               the delayed functor and the args
         * @param functor the callable to be wrapped
         * @param worker must outlive the actual execution
         */
        DelayingFunctor(Pusher const& pusher, Functor const& functor, Worker & worker)
          :  pusher(pusher), functor(functor), worker(worker) {}

        /** Bind args to functor and pass the
         * resulting nullary functor to pusher.
         *
         * @return future of functors return
         */
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

