
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
    , public Scheduler::ProtoProperties
    , public Scheduler::RuntimeProperties
{
public:
    Schedulable( Scheduler & scheduler_ )
        : scheduler( scheduler_ )
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
    typename Policy::ProtoProperty &
    proto_property( void )
    {
        return scheduler.template proto_property< Policy >( *this );
    }

    template< typename Policy >
    typename Policy::RuntimeProperty &
    runtime_property( void )
    {
        return scheduler.template runtime_property< Policy >( *this );
    }

private:
    Schedulable * last;
    Scheduler & scheduler;
};

} // namespace rmngr
