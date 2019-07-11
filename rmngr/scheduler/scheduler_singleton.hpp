
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
    using Task = typename SchedulerType::Task;
    using Properties = typename SchedulerType::Properties;
    using PropertiesPatch = typename SchedulerType::PropertiesPatch;

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

    template < typename NullaryCallable >
    static void emplace_task( NullaryCallable && impl, Properties const & prop )
    {
        getInstance().emplace_task( std::move(impl), prop );
    }

    template < typename ImplCallable, typename PropCallable >
    static auto make_functor( ImplCallable && impl, PropCallable && prop )
    {
        return getInstance().make_functor( std::move(impl), std::move(prop) );
    }

    static auto update_properties( PropertiesPatch const & patch )
    {
        return getInstance().update_properties( patch );
    }

}; // class SchedulerSingleton

} // namespace rmngr

