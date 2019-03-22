
/**
 * @file rmngr/scheduler/schedulable_functor.hpp
 */

#pragma once

#include <rmngr/scheduler/schedulable.hpp>

namespace rmngr
{

template<
    typename Scheduler,
    typename DelayedFunctor
>
struct SchedulableFunctor
    : public DelayedFunctor
    , public Schedulable< Scheduler >
{
    SchedulableFunctor(
	DelayedFunctor && f,
        typename Scheduler::ProtoProperties const & props,
        Scheduler & scheduler
    )
        : DelayedFunctor( std::forward< DelayedFunctor >( f ) )
        , Schedulable< Scheduler >( scheduler )
    {
        typename Scheduler::ProtoProperties & p = *this;
        p = props;
    }
}; // struct SchedulableFunctor

template<
    typename Scheduler,
    typename Functor
>
class ProtoSchedulableFunctor
    : public Scheduler::ProtoProperties
{
public:
    ProtoSchedulableFunctor( Functor const & f, Scheduler & scheduler_ )
        : functor( f )
        , scheduler( scheduler_ )
    {
    }

    template<
        typename DelayedFunctor,
        typename... Args
    >
    SchedulableFunctor< Scheduler, DelayedFunctor > *
    clone( DelayedFunctor && f, Args &&... args ) const
    {
        return new SchedulableFunctor< Scheduler, DelayedFunctor >(
            std::forward< DelayedFunctor >( f ), *this, this->scheduler );
    }

    template< typename... Args >
    typename std::result_of< Functor( Args... ) >::type
    operator()( Args &&... args )
    {
        return this->functor( std::forward< Args >( args )... );
    }

private:
    Functor functor;
    Scheduler & scheduler;
}; // class ProtoSchedulableFunctor

template<
    typename Scheduler,
    typename Functor,
    typename PropertyFun
>
class PreparingProtoSchedulableFunctor
    : public ProtoSchedulableFunctor< Scheduler, Functor >
{
public:
    PreparingProtoSchedulableFunctor(
        Functor const & f,
        Scheduler & scheduler,
        PropertyFun const & prepare_properties_
    )
        : ProtoSchedulableFunctor< Scheduler, Functor >( f, scheduler )
        , prepare_properties( prepare_properties_ )
    {
    }

    template<
        typename DelayedFunctor,
        typename... Args
    >
    SchedulableFunctor< Scheduler, DelayedFunctor > *
    clone(
        DelayedFunctor && f,
	Args &&... args
    ) const
    {
        SchedulableFunctor< Scheduler, DelayedFunctor > * schedulable =
	    this->ProtoSchedulableFunctor< Scheduler, Functor >::clone(
                std::forward< DelayedFunctor >( f ),
                std::forward< Args >( args )... );

        this->prepare_properties(
            schedulable, std::forward< Args >( args )... );

        return schedulable;
    }

private:
    PropertyFun prepare_properties;
};


} // namespace rmngr
