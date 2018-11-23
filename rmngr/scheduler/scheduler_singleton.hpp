
/**
 * @file rmngr/scheduler/scheduler_singleton.hpp
 */

#pragma once

#include <utility> // std::forward
#include <rmngr/scheduler/scheduler.hpp>

namespace rmngr
{

template <
    typename SchedulingPolicies = boost::mpl::vector<>,
    template <typename Graph> class Refinement = DefaultRefinement,
    template <typename T> class Graph = DefaultGraph
>
class SchedulerSingleton
{
private:
    using SchedulerType =
        Scheduler<
            SchedulingPolicies,
            Refinement,
            Graph
        >;

    static SchedulerType* & getPtr()
    {
        static SchedulerType * ptr;
        return ptr;
    }

public:
    using Schedulable = typename SchedulerType::Schedulable;
    using SchedulablePtr = observer_ptr< Schedulable >;

    static void init( int nthreads )
    {
        getPtr() = new SchedulerType( nthreads );
    }

    static void finish( void )
    {
        delete getPtr();
    }

    static SchedulerType & getInstance()
    {
        return *(getPtr());
    }

#define CALL_INSTANCE( name, args ) \
    getInstance(). name ( args )

#define WRAP_INSTANCE( name ) \
    template < typename... Args > \
    static auto name ( Args&&... args ) \
      -> decltype( CALL_INSTANCE( name, std::forward<Args>(args)... ) ) \
    { \
        return CALL_INSTANCE( name, std::forward<Args>(args)... ); \
    } \

    WRAP_INSTANCE( make_proto )
    WRAP_INSTANCE( make_functor )
    WRAP_INSTANCE( get_current_queue )

    /**
     * Create a functor and enqueue it immediately
     * in the current refinement
     */
    template <
        typename Functor,
        typename PropertyFun
    >
    static auto enqueue_functor(
        Functor const & impl,
        PropertyFun const & property_fun
    )
    {
        auto functor = make_functor( impl, property_fun );
        return functor();
    }
}; // class SchedulerSingleton

} // namespace rmngr

