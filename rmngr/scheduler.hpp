
/**
 * @file rmngr/scheduler.hpp
 */

#pragma once

#include <boost/mpl/inherit.hpp>
#include <boost/mpl/inherit_linearly.hpp>
#include <boost/mpl/for_each.hpp>

#include <rmngr/functor.hpp>
#include <rmngr/scheduling_graph.hpp>


#include <boost/graph/adjacency_list.hpp>
#include <rmngr/precedence_graph.hpp>
#include <rmngr/resource_user.hpp>

namespace rmngr
{

template <typename T>
using DefaultGraph =
boost::adjacency_list<
    boost::setS,
    boost::vecS,
    boost::bidirectionalS,
    T
>;

template <typename Graph>
using DefaultRefinement =
QueuedPrecedenceGraph<
    Graph,
    ResourceUser
>;

/**
 * Compose the Scheduler from multiple Scheduling-Policies.
 *
 * @tparam SchedulingPolicies Model of boost::mpl Forward-Sequence
 * @tparam Refinement Refinement type for Precedence-Graph
 * @tparam Graph Graph<T> is a complete boost::graph type
 */
template <
    typename SchedulingPolicies,
    template <typename Graph> class Refinement = DefaultRefinement,
    template <typename T> class Graph = DefaultGraph
>
class Scheduler
{
private:
    template < typename Policy >
    struct SchedulingProperty
    {
        typename Policy::Property prop;
    };

    using SchedulingProperties =
        typename boost::mpl::inherit_linearly<
            SchedulingPolicies,
            boost::mpl::inherit< boost::mpl::_1, SchedulingProperty<boost::mpl::_2> >
        >::type;

public:
    Scheduler()
       : graph( main_refinement )
    {}

    /**
     * Base class storing all scheduling info and the functor
     */
    struct Schedulable
        : public SchedulingProperties
        , virtual public DelayedFunctorInterface
    {};

    template <typename DelayedFunctor>
    struct SchedulableFunctor
        : public DelayedFunctor
        , public Schedulable
    {
        SchedulableFunctor( DelayedFunctor && f, SchedulingProperties const & props )
            : DelayedFunctor( std::forward<DelayedFunctor>( f ) )
            , SchedulingProperties( props )
        {}
    }; // struct SchedulableFunctor

    template <typename Functor>
    class ProtoSchedulableFunctor
        : public SchedulingProperties
    {
      public:
        ProtoSchedulableFunctor(
            Functor const & f,
            SchedulingProperties const & props
        )
            : SchedulingProperties( props )
            , functor( f )
        {}

        template <typename DelayedFunctor>
        SchedulableFunctor<DelayedFunctor> *
        clone( DelayedFunctor && f ) const
        {
            return new SchedulableFunctor<DelayedFunctor>(
                std::forward<DelayedFunctor>( f ), *this );
        }

        template <typename... Args>
        typename std::result_of<Functor( Args... )>::type
        operator()( Args &&... args )
        {
            return this->functor( std::forward<Args>( args )... );
        }

      private:
        Functor functor;
    }; // class ProtoSchedulableFunctor

    void update(void)
    {
        while ( this->graph.is_deprecated() )
        {
            if ( this->graph_mutex.try_lock() )
            {
                this->graph.update();

                Updater updater{ *this };
                boost::mpl::for_each< SchedulingPolicies >( updater );
            }
        }
    }

    template <typename Policy>
    Policy & policy( void )
    {
        return this->policies;
    }

private:
    std::mutex graph_mutex;
    SchedulingGraph< Graph<observer_ptr<Schedulable>> > graph;
    Refinement< Graph<observer_ptr<Schedulable>> > main_refinement;

    struct Updater
    {
        Scheduler & scheduler;

        template <typename Policy>
        void operator() (Policy & policy)
        {
            policy.update( scheduler.graph );
        }
    };

    typename boost::mpl::inherit_linearly<
        SchedulingPolicies,
        boost::mpl::inherit< boost::mpl::_1, boost::mpl::_2 >
    >::type
    policies;

}; // class Scheduler

} // namespace rmngr

