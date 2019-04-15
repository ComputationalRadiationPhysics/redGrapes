
/**
 * @file rmngr/scheduler/scheduler.hpp
 */

#pragma once

#include <condition_variable>
#include <functional> // std::reference_wrapper
#include <boost/mpl/inherit.hpp>
#include <boost/mpl/inherit_linearly.hpp>
#include <boost/mpl/for_each.hpp>

#include <rmngr/functor.hpp>
#include <rmngr/functor_queue.hpp>
#include <rmngr/thread_dispatcher.hpp>
#include <rmngr/scheduler/scheduler_interface.hpp>
#include <rmngr/scheduler/schedulable.hpp>
#include <rmngr/scheduler/schedulable_functor.hpp>
#include <rmngr/scheduler/scheduling_graph.hpp>

// defaults
#include <boost/graph/adjacency_list.hpp>
#include <rmngr/graph/precedence_graph.hpp>
#include <rmngr/resource/resource_user.hpp>

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

template <typename T>
struct DefaultEnqueuePolicy
{
    static bool is_serial(T const & a, T const & b) { return true; }
    static bool is_superset(T const & super, T const & sub) { return false; };
};

template <typename Graph>
using DefaultRefinement =
QueuedPrecedenceGraph<
    Graph,
    DefaultEnqueuePolicy<typename Graph::ID>
>;

struct DefaultSchedulingPolicy
{
    struct ProtoProperty {};
    struct RuntimeProperty {};

    void init( SchedulerInterface & ) {}
    void finish() {}
    void notify() {}

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
public:
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

    friend class rmngr::Schedulable<Scheduler>;
    using Schedulable = rmngr::Schedulable<Scheduler>;

    template <typename DelayedFunctor>
    using SchedulableFunctor = SchedulableFunctor<Scheduler, DelayedFunctor>;

    template <typename Functor>
    using ProtoSchedulableFunctor = ProtoSchedulableFunctor<Scheduler, Functor>;

    template <typename Functor, typename PropertyFun>
    using PreparingProtoSchedulableFunctor = PreparingProtoSchedulableFunctor<Scheduler, Functor, PropertyFun>;

    Scheduler( size_t nthreads = 1 )
      : graph( &uptodate, main_refinement )
      , currently_scheduled( nthreads+1 )
      , uptodate( *this )
    {
        this->for_each_policy< PolicyInit >();
    }

    ~Scheduler()
    {
        this->for_each_policy< PolicyFinish >();
    }

    size_t num_threads(void) const
    {
        return this->currently_scheduled.size()-1;
    }

    template < typename Functor >
    ProtoSchedulableFunctor< Functor >
    make_proto( Functor const & f )
    {
        return ProtoSchedulableFunctor< Functor >( f, *this );
    }

    template <
        typename Functor,
        typename PropertyFun
    >
    PreparingProtoSchedulableFunctor<Functor, PropertyFun>
    make_proto(
        Functor const & f,
        PropertyFun const & prepare_properties
    )
    {
        return
        PreparingProtoSchedulableFunctor<
            Functor,
            PropertyFun
        >(
            f,
            *this,
            prepare_properties
        );
    }

public:
    struct UpToDateFlag
      : std::atomic_flag
      , virtual FlagInterface
    {
        Scheduler & scheduler;
        UpToDateFlag( Scheduler& scheduler )
	    : std::atomic_flag(ATOMIC_FLAG_INIT)
            , scheduler(scheduler)
        {}

        void clear()
        {
	    this->std::atomic_flag::clear();
	    // notify all policies
	    this->scheduler.for_each_policy< PolicyNotify >();
	}
    };

    UpToDateFlag uptodate;

    void update(void)
    {
        auto lock = this->lock();
        while( ! this->uptodate.test_and_set() )
        {
            this->graph.update();
	    this->for_each_policy< PolicyUpdate >();
        }
    }

    template <typename Policy>
    Policy & policy( void )
    {
        return this->policies;
    }

    bool empty(void)
    {
        auto lock = this->lock();
        return this->graph.empty();
    }

    FunctorQueue< Refinement< Graph<Schedulable*> >, WorkerInterface >
    get_main_queue( void )
    {
        return make_functor_queue( this->main_refinement, *this->worker, this->mutex );
    }

    Schedulable *
    get_current_schedulable( void )
    {
        return this->currently_scheduled[thread::id];
    }

    template <
        typename SRefinement = Refinement< Graph<Schedulable*> >
    >
    SRefinement &
    get_current_refinement( void )
    {
        std::lock_guard<std::mutex> lock( this->mutex );
        if( this->get_current_schedulable() )
        {
            auto r = this->main_refinement.template refinement<SRefinement>(
                       this->get_current_schedulable()
                   );

            if(r)
                return *r;
        }
        return this->main_refinement;
    }

    template <
        typename SRefinement = Refinement< Graph<Schedulable*> >
    >
    FunctorQueue< SRefinement, WorkerInterface >
    get_current_queue( void )
    {
        return make_functor_queue(
                   this->get_current_refinement< SRefinement >(),
                   *this->worker,
                   this->mutex
               );
    }

    struct CurrentQueuePusher
    {
        Scheduler * scheduler;

        template <
            typename ProtoFunctor,
            typename DelayedFunctor,
            typename... Args
        >
        void operator() (
            ProtoFunctor const& proto,
            DelayedFunctor&& delayed,
            Args&&... args
        )
        {
            auto& queue = scheduler->get_current_refinement();
            auto lock = scheduler->lock();

            queue.push(
	        proto.clone(
		    std::forward<DelayedFunctor>(delayed),
                    std::forward<Args>(args)...
	    ));
        }
    };

    template < typename Functor >
    using CurrentQueueFunctor = DelayingFunctor< CurrentQueuePusher, Functor, WorkerInterface >;

    template <typename ProtoFunctor>
    auto make_functor( ProtoFunctor const & proto )
    {
        return make_delaying( CurrentQueuePusher{ this }, proto, *this->worker );
    }

    template <typename Functor, typename PropertyFun>
    auto make_functor( Functor const& f, PropertyFun const & prop )
    {
        return make_functor( this->make_proto( f, prop ) );
    }

    template<
        typename Policy,
        typename... Args
    >
    void update_property( Args&&... args )
    {
        this->lock();
        auto s = this->get_current_schedulable();
	if(!s)
            throw std::runtime_error("invalid update_property: no schedulable running");

        this->policy< Policy >().update_property(*s, *s, std::forward<Args>(args)...);

        auto ref = dynamic_cast< Refinement<Graph<Schedulable*>>* >(
                       this->main_refinement.find_refinement_containing( s ));
        ref->update_vertex( s );

        this->update();
    }

private:
    Refinement< Graph<Schedulable*> > main_refinement;
    SchedulingGraph< Graph<Schedulable*> > graph;
    std::vector< Schedulable* > currently_scheduled;

    typename boost::mpl::inherit_linearly<
        SchedulingPolicies,
        boost::mpl::inherit< boost::mpl::_1, boost::mpl::_2 >
    >::type policies;

#define POLICY_FUNCTOR( NAME, CALL )             \
    struct NAME                                  \
    {                                            \
        Scheduler & scheduler;                   \
                                                 \
        template< typename Policy >              \
        void operator()( boost::type< Policy > ) \
        {                                        \
            scheduler.policy< Policy >().CALL;   \
        }                                        \
    };

    POLICY_FUNCTOR( PolicyInit, init(scheduler) )
    POLICY_FUNCTOR( PolicyUpdate, update(scheduler.graph, scheduler)  )
    POLICY_FUNCTOR( PolicyFinish, finish() )
    POLICY_FUNCTOR( PolicyNotify, notify() )

    template <typename PolicyFun>
    void for_each_policy()
    {
        PolicyFun f{ *this };
	boost::mpl::for_each<
	    SchedulingPolicies,
	    boost::type<boost::mpl::_>
	>( f );
    }

}; // class Scheduler

} // namespace rmngr

