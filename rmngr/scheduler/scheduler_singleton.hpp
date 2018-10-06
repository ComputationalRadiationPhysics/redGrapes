
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
    WRAP_INSTANCE( get_current_queue )

    /**
     * Create a functor in the current refinement
     * and enqueue it immediately
     */
    template <
        typename Functor,
        typename PropertyFun
    >
    static void enqueue_functor(
        Functor const & impl,
        PropertyFun const & property_fun
    )
    {
        auto queue = get_current_queue();
        auto functor = queue.make_functor(
            make_proto( impl, property_fun )
        );

        functor();
    }
}; // class SchedulerSingleton

} // namespace rmngr

