
/**
 * @file rmngr/scheduler/scheduler.hpp
 */

#pragma once

#include <boost/mpl/inherit.hpp>
#include <boost/mpl/inherit_linearly.hpp>
#include <boost/mpl/for_each.hpp>

#include <rmngr/functor.hpp>
#include <rmngr/functor_queue.hpp>
#include <rmngr/thread_dispatcher.hpp>
#include <rmngr/scheduler/scheduling_graph.hpp>

// defaults
#include <boost/graph/adjacency_list.hpp>
#include <rmngr/graph/precedence_graph.hpp>
#include <rmngr/resource/resource_user.hpp>

namespace rmngr
{

struct SchedulerInterface
{
    struct SchedulableInterface
        : virtual public DelayedFunctorInterface
    {
        virtual void start(void) = 0;
        virtual void finish(void) = 0;
    };

    virtual void update(void) = 0;
    virtual bool empty(void) = 0;

    std::unique_lock< std::mutex >
    lock(void)
    {
        return std::unique_lock<std::mutex>(this->mutex);
    };

    template < typename Policy >
    struct ProtoProperty
    {
        typename Policy::ProtoProperty prop;
        operator typename Policy::ProtoProperty& ()
        { return this->prop; }
    };

    template < typename Policy >
    struct RuntimeProperty
    {
        typename Policy::RuntimeProperty prop;
        operator typename Policy::RuntimeProperty& ()
        { return this->prop; }
    };

    template < typename Policy >
    static typename Policy::ProtoProperty &
    proto_property( ProtoProperty<Policy> & s )
    { return s.prop; }

    template < typename Policy >
    static typename Policy::RuntimeProperty &
    runtime_property( RuntimeProperty<Policy> & s )
    { return s.prop; }

protected:
    std::mutex mutex;
};

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

struct DefaultSchedulingPolicy
{
    struct ProtoProperty {};
    struct RuntimeProperty {};

    template <typename Graph>
    void update( Graph & graph, SchedulerInterface & scheduler ) {}
};

/**
 * Compose the Scheduler from multiple Scheduling-Policies.
 *
 * @tparam SchedulingPolicies Model of boost::mpl Forward-Sequence
 * @tparam Refinement Refinement type for Precedence-Graph
 * @tparam Graph Graph<T> is a complete boost::graph type
 */
template <
    typename SchedulingPolicies = boost::mpl::vector<>,
    template <typename Graph> class Refinement = DefaultRefinement,
    template <typename T> class Graph = DefaultGraph
>
class Scheduler
  : public SchedulerInterface
{
private:
    using ProtoProperties =
        typename boost::mpl::inherit_linearly<
            SchedulingPolicies,
            boost::mpl::inherit< boost::mpl::_1, ProtoProperty<boost::mpl::_2> >
        >::type;

    using RuntimeProperties =
        typename boost::mpl::inherit_linearly<
            SchedulingPolicies,
            boost::mpl::inherit< boost::mpl::_1, RuntimeProperty<boost::mpl::_2> >
        >::type;

public:
    Scheduler( int nthreads )
      : graph( main_refinement ),
        currently_scheduled( nthreads+1 )
    {}

    /**
     * Base class storing all scheduling info and the functor
     */
    struct Schedulable
        : public virtual SchedulerInterface::SchedulableInterface
        , public ProtoProperties
        , public RuntimeProperties
    {
        Schedulable( Scheduler & scheduler_ )
            : scheduler(scheduler_) {}

        void start(void)
        {
            this->scheduler.currently_scheduled[ thread::id ] = this;
        }

        void finish(void)
        {
            this->scheduler.finish( this );
        }

    private:
        Scheduler & scheduler;
    };

    template <typename DelayedFunctor>
    struct SchedulableFunctor
        : public DelayedFunctor
        , public Schedulable
    {
        SchedulableFunctor(
            DelayedFunctor && f,
            ProtoProperties const & props,
            Scheduler & scheduler
        )
            : DelayedFunctor( std::forward<DelayedFunctor>( f ) )
            , Schedulable( scheduler )
        {
            ProtoProperties& p = *this;
            p = props;
        }
    }; // struct SchedulableFunctor

    template <typename Functor>
    class ProtoSchedulableFunctor
        : public ProtoProperties
    {
    public:
        ProtoSchedulableFunctor(
            Functor const & f,
            Scheduler & scheduler_
        )
            : functor( f )
            , scheduler(scheduler_)
        {}

        template <typename DelayedFunctor>
        SchedulableFunctor<DelayedFunctor> *
        clone( DelayedFunctor && f ) const
        {
            return new SchedulableFunctor<DelayedFunctor>(
                std::forward<DelayedFunctor>( f ),
                *this,
                this->scheduler
            );
        }

        template <typename... Args>
        typename std::result_of<Functor( Args... )>::type
        operator()( Args &&... args )
        {
            return this->functor( std::forward<Args>( args )... );
        }

    private:
        Functor functor;
        Scheduler & scheduler;
    }; // class ProtoSchedulableFunctor

    template <typename Functor>
    ProtoSchedulableFunctor<Functor>
    make_proto( Functor const & f )
    {
        return ProtoSchedulableFunctor<Functor>( f, *this );
    }

    void finish( observer_ptr< Schedulable > s )
    {
        if( this->graph.finish(s) )
            delete &s;
    }

    void update(void)
    {
        while ( this->graph.is_deprecated() )
        {
            if ( this->mutex.try_lock() )
            {
                this->graph.update();

                Updater updater{ *this };
                boost::mpl::for_each<
                    SchedulingPolicies,
                    boost::type<boost::mpl::_>
                >( updater );

                this->mutex.unlock();
            }
        }
    }

    template <typename Policy>
    Policy & policy( void )
    {
        return this->policies;
    }

    bool empty(void)
    {
        this->update();
        auto lock = this->lock();
        return this->graph.empty();
    }

    FunctorQueue< Refinement< Graph< observer_ptr<Schedulable> > > >
    get_main_queue( void )
    {
        return make_functor_queue( this->main_refinement, this->mutex );
    }

    observer_ptr<Schedulable>
    get_current_schedulable( void )
    {
        return this->currently_scheduled[thread::id];
    }

    template <typename SRefinement>
    observer_ptr<SRefinement>
    get_current_refinement( void )
    {
        std::lock_guard<std::mutex> lock( this->mutex );
        return this->main_refinement.template refinement<SRefinement>(
                   this->get_current_schedulable()
               );
    }

    template <
        typename SRefinement = Refinement< Graph<observer_ptr<Schedulable>> >
    >
    FunctorQueue< SRefinement >
    get_current_queue( void )
    {
        auto refinement = this->get_current_refinement< SRefinement >();
        return make_functor_queue( *refinement, this->mutex );
    }

private:
    Refinement< Graph<observer_ptr<Schedulable>> > main_refinement;
    SchedulingGraph< Graph<observer_ptr<Schedulable>> > graph;
    std::vector< observer_ptr<Schedulable> > currently_scheduled;

    typename boost::mpl::inherit_linearly<
        SchedulingPolicies,
        boost::mpl::inherit< boost::mpl::_1, boost::mpl::_2 >
    >::type policies;

    struct Updater
    {
        Scheduler & scheduler;

        template <typename Policy>
        void operator() ( boost::type<Policy> )
        {
            scheduler.policy<Policy>().update( scheduler.graph, scheduler );
        }
    };

}; // class Scheduler

} // namespace rmngr

