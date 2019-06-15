
/**
 * @file rmngr/scheduler/scheduler.hpp
 */

#pragma once

#include <condition_variable>
#include <functional> // std::reference_wrapper
#include <boost/mpl/inherit.hpp>
#include <boost/mpl/inherit_linearly.hpp>
#include <boost/mpl/for_each.hpp>

#include <rmngr/delayed_functor.hpp>
#include <rmngr/working_future.hpp>
#include <rmngr/thread_dispatcher.hpp>
#include <rmngr/scheduler/scheduler_interface.hpp>
#include <rmngr/scheduler/schedulable.hpp>
#include <rmngr/scheduler/scheduling_graph.hpp>

// defaults
#include <boost/graph/adjacency_list.hpp>
#include <rmngr/graph/precedence_graph.hpp>

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
    static void assert_superset(T const & super, T const & sub) {}
};

template <typename Graph>
using DefaultRefinement =
QueuedPrecedenceGraph<
    Graph,
    DefaultEnqueuePolicy
>;

struct DefaultSchedulingPolicy
{
    struct Property
    {
        struct Patch {};
        void apply_patch( Patch const & patch ) {}
    };

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
    using Properties =
        typename boost::mpl::inherit_linearly<
            SchedulingPolicies,
            boost::mpl::inherit< boost::mpl::_1, Property<boost::mpl::_2> >
        >::type;
    /*
    struct PropertyPatch :
        boost::mpl::inherit_linearly<
            SchedulingPolicies,
            boost::mpl::inherit< boost::mpl::_1, PropertyPatch<boost::mpl::_2> >
        >::type
    {
        struct AddPatch
        {
            PropertyPatch & first;
            PropertyPatch const & second;

            template< typename Policy >
            void operator()( boost::type< Policy > )
            {
                Policy::Property::Patch & p1 = first;
                Policy::Property::Patch & p2 = second;
                p1 += p2;
            }
        };

        void operator+= ( PropertyPatch const & other )
        {
            boost::mpl::for_each<
                SchedulingPolicies,
                boost::type<boost::mpl::_>
            >( AddPatch{ *this, other } );
        }
    }
    */
    friend class rmngr::Schedulable<Scheduler>;
    using Schedulable = rmngr::Schedulable<Scheduler>;

    Scheduler( size_t nthreads = 1 )
      : uptodate( *this )
      , graph( uptodate, main_refinement )
      , currently_scheduled( nthreads+1 )
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
        while( ! this->uptodate.test_and_set() )
        {
	    auto lock = this->lock();
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
        auto lock = this->lock();
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

    /**
     * generate the trace of parent tasks for the current task
     */
    std::experimental::optional<std::vector<Schedulable*>> backtrace()
    {
        return this->main_refinement.backtrace( this->get_current_schedulable() );
    }

    /**
     * Create a task and enqueue it immediately
     */
    template< typename NullaryCallable >
    void make_task( NullaryCallable && impl, Properties const & prop = Properties{} )
    {
        this->push( new SchedulableFunctor<Scheduler, NullaryCallable>(std::move(impl), prop, *this) );
    }

    template< typename ImplCallable, typename PropCallable >
    struct TaskFactoryFunctor
    {
        Scheduler & scheduler;
        ImplCallable impl;
        PropCallable prop;

        template <typename... Args>
        auto operator() (Args&&... args)
        {
            Properties props = this->prop( std::forward<Args>(args)... );
            auto applied = std::bind( this->impl, std::forward<Args>(args)... );
            auto delayed = make_delayed_functor( std::move(applied) );
            auto result = make_working_future( delayed.get_future(), *scheduler.worker );
            scheduler.make_task( std::move(delayed), props );
            return result;
        }
    };

    template< typename ImplCallable, typename PropCallable >
    auto make_functor(ImplCallable && impl, PropCallable && prop)
    {
        return TaskFactoryFunctor<ImplCallable, PropCallable>{*this, impl, prop};
    }

    /**
     * Enqueue a Schedulable as child of the current task.
     */
    void push( Schedulable * s )
    {
        auto lock = this->lock();
        this->get_current_refinement().push( s );
    }

    /**
     * Apply a patch to the properties of the current schedulable
     *//*
    void update_property( PropertyPatch const & patch )
    {
        update_property( get_current_schedulable(), patch );
        }*/

    /**
     * Apply a patch to the properties of a schedulable and
     * recalculate its dependencies
     *
     * @param s Schedulable to be updated
     * @param patch changes on the properties
     *//*
    void update_property( Schedulable * s, PropertyPatch const & patch )
    {
	if(!s)
            throw std::runtime_error("update_property: invalid schedulable");

        auto lock = this->lock();
	boost::mpl::for_each<
	    SchedulingPolicies,
	    boost::type<boost::mpl::_>
            >( PropertyPatcher{ *s, patch, *this } );

        auto ref = dynamic_cast< Refinement<Graph<Schedulable*>>* >(
                       this->main_refinement.find_refinement_containing( s ));
        ref->update_vertex( s );

        this->update();
        }*/

private:
    Refinement< Graph<Schedulable*> > main_refinement;
    SchedulingGraph< Graph<Schedulable*> > graph;
    std::vector< Schedulable* > currently_scheduled;

    typename boost::mpl::inherit_linearly<
        SchedulingPolicies,
        boost::mpl::inherit< boost::mpl::_1, boost::mpl::_2 >
    >::type policies;

    struct PropertyPatcher
    {
        Schedulable & s;
        //PropertyPatch const & patch;
        Scheduler & scheduler;

        template< typename Policy >
        void operator() ( boost::type< Policy > )
        {
            // s.template property< Policy >().apply_patch( patch );
        }
    };

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

