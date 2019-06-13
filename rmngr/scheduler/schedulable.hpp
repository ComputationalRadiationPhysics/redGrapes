
/**
 * @file rmngr/scheduler/schedulable.hpp
 */

#pragma once

namespace rmngr
{

/**
 * Base class storing all scheduling info and the functor
 */
template<
    typename Scheduler
>
class Schedulable
    : public virtual Scheduler::SchedulableInterface
{
public:
    Schedulable( typename Scheduler::Properties const & prop,  Scheduler & scheduler_ )
        : scheduler( scheduler_ )
        , properties( prop )
	, last( nullptr )
    {
    }

    void
    start( void )
    {
        last = this->scheduler.currently_scheduled[ thread::id ];
        this->scheduler.currently_scheduled[ thread::id ] = this;
    }

    void
    end( void )
    {
        this->scheduler.currently_scheduled[ thread::id ] = this->last;
    }

    void
    finish( void )
    {
        if( this->scheduler.graph.finish( this ) )
            delete this;
    }

    template< typename Policy >
    typename Policy::Property &
    property( void )
    {
        return this->properties;
    }

private:
    Schedulable * last;
    Scheduler & scheduler;

    typename Scheduler::Properties properties;
}; // class Schedulable

template<
    typename Scheduler,
    typename NullaryCallable
>
struct SchedulableFunctor
    : public Schedulable< Scheduler >
{
    SchedulableFunctor(
	NullaryCallable && impl_,
        typename Scheduler::Properties const & properties,
        Scheduler & scheduler
    )
        : Schedulable< Scheduler >( properties, scheduler )
        , impl( std::move(impl_) )
    {}

    void run()
    {
        this->impl();
    }

private:
    NullaryCallable impl;
}; // struct SchedulableFunctor

} // namespace rmngr
