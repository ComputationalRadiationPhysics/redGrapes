#pragma once

#include <functional>

namespace rmngr
{

struct DelayedFunctorInterface
{
    virtual ~DelayedFunctorInterface() {};
    virtual void run (void) = 0;
}; // class DelayedFunctorInterface

template <typename Pusher, typename Functor>
class DelayingFunctor
{
    private:
        template <typename AppliedFunctor>
        class DelayedFunctor : virtual public DelayedFunctorInterface, public AppliedFunctor
        {
            public:
                DelayedFunctor(AppliedFunctor const& f)
                    : AppliedFunctor(f)
                {}

                void run (void)
                {
                    (*this)();
                }
        };

        template <typename AppliedFunctor>
        static DelayedFunctor<AppliedFunctor> make_delayed_functor(AppliedFunctor const& f)
        {
            return DelayedFunctor<AppliedFunctor>(f);
        }

        Pusher pusher;
        Functor functor;

    public:
        DelayingFunctor(Pusher const& pusher_, Functor const& functor_)
            :  pusher(pusher_), functor(functor_)
        {}

        template <typename... Args>
        void operator() (Args&&... args)
        {
            this->pusher(this->functor, make_delayed_functor(std::bind(this->functor, args...)));
        }
}; // class DelayingFunctor

template <typename Pusher, typename Functor>
DelayingFunctor<Pusher, Functor> make_delaying(Pusher const& p, Functor const& f)
{
    return DelayingFunctor<Pusher, Functor>(p, f);
}

} // namespace rmngr

