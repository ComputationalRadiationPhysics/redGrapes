
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include <rmngr/thread_dispatcher.hpp>
#include <rmngr/scheduling_graph.hpp>
#include <rmngr/delayed_functor.hpp>
#include <rmngr/working_future.hpp>
#include <rmngr/graph/refined_graph.hpp>
#include <rmngr/graph/precedence_graph.hpp>
#include <rmngr/graph/util.hpp>

#include <rmngr/scheduler/fifo.hpp>

namespace rmngr
{

struct DefaultTaskProperties
{
    struct Patch {};
};

template <typename T>
struct DefaultEnqueuePolicy
{
    static bool is_serial(T const & a, T const & b) { return true; }
    static void assert_superset(T const & super, T const & sub) {}
};

template <
    typename TaskProperties = DefaultTaskProperties,
    typename EnqueuePolicy = DefaultEnqueuePolicy< TaskProperties >,
    template <typename> class Scheduler = FIFOScheduler
>
class Manager
{
public:
    struct Task
    {
        Task( TaskProperties const & properties )
            : properties( properties )
            , before_hook( []{} )
            , after_hook( []{} )
        {}

        virtual ~Task() = default;

        virtual void run() = 0;

        void operator() ()
        {
            before_hook();
            run();
            after_hook();
        }

        void hook_before( std::function<void(void)> const & hook )
        {
            before_hook = [rest=std::move(before_hook), hook]{ hook(); rest(); };
        }

        void hook_after( std::function<void(void)> const & hook )
        {
            after_hook = [rest=std::move(after_hook), hook]{ rest(); hook(); };
        }

        std::function<void(void)> before_hook, after_hook;
        TaskProperties properties;
    };

    template< typename NullaryCallable >
    struct FunctorTask : Task
    {
        FunctorTask( NullaryCallable && impl, TaskProperties const & properties )
            : Task( properties )
            , impl( std::move(impl) )
        {}

        void run()
        {
            this->impl();
        }

    private:
        NullaryCallable impl;
    };

    struct TaskEnqueuePolicy
    {
        static bool is_serial(Task * a, Task * b) { return EnqueuePolicy::is_serial(a->properties, b->properties); }
        static void assert_superset(Task * super, Task * sub) { EnqueuePolicy::assert_superset(super->properties, sub->properties); }
    };

    using Refinement = QueuedPrecedenceGraph<
        boost::adjacency_list<
            boost::setS,
            boost::vecS,
            boost::bidirectionalS,
            Task*
        >,
        TaskEnqueuePolicy
    >;

    struct Worker
    {
        SchedulingGraph< Task > & scheduling_graph;

        void operator() ( std::function<bool()> const& pred )
        {
             while( !pred() )
                 scheduling_graph.consume_job( pred );
        }
    };

    Refinement precedence_graph;
    SchedulingGraph< Task > scheduling_graph;
    ThreadDispatcher< SchedulingGraph<Task> > thread_dispatcher;
    Scheduler< SchedulingGraph<Task> > scheduler;
    Worker worker;

public:
    Manager( int n_threads = 2 )
        : scheduling_graph( precedence_graph, n_threads )
        , thread_dispatcher( scheduling_graph, n_threads )
        , scheduler( scheduling_graph )
        , worker{ scheduling_graph }
    {}

    ~Manager()
    {
        scheduling_graph.finish();
        thread_dispatcher.finish();
    }

    template< typename NullaryCallable >
    auto emplace_task( NullaryCallable && impl, TaskProperties const & prop = TaskProperties{} )
    {
        auto delayed = make_delayed_functor( std::move(impl) );
        auto result = make_working_future( std::move(delayed.get_future()), this->worker );
        this->push( new FunctorTask< decltype(delayed) >( std::move(delayed), prop ) );
        return result;
    }

    /**
     * Enqueue a Schedulable as child of the current task.
     */
    void push( Task * task )
    {
        this->get_current_refinement().push( task );
        scheduler.new_task( task );
    }

    Refinement &
    get_current_refinement( void )
    {
        if( std::experimental::optional< Task* > task = scheduling_graph.get_current_task() )
        {
            auto r = this->precedence_graph.template refinement<Refinement>( *task );

            if(r)
                return *r;
        }

        return this->precedence_graph;
    }

    void update_properties( typename TaskProperties::Patch const & patch )
    {
        if( std::experimental::optional< Task* > task = scheduling_graph.get_current_task() )
            update_properties( *task, patch );
        else
            throw std::runtime_error("update_properties: currently no task running");
    }

    void update_properties( Task * task, typename TaskProperties::Patch const & patch )
    {
        task->properties.apply_patch( patch );
        scheduling_graph.template update_vertex< Refinement >( task );
    }

    auto backtrace()
    {
        if( std::experimental::optional< Task* > task = scheduling_graph.get_current_task() )
            return precedence_graph.backtrace( *task );
        else
            return std::experimental::nullopt;
    }

    template< typename ImplCallable, typename PropCallable >
    struct TaskFactoryFunctor
    {
        Manager & mgr;
        ImplCallable impl;
        PropCallable prop;

        template <typename... Args>
        auto operator() (Args&&... args)
        {
            return mgr.emplace_task(
                       std::bind( this->impl, std::forward<Args>(args)... ),
                       this->prop( std::forward<Args>(args)... )
                   );
        }
    };

    struct DefaultPropFunctor
    {
        template < typename... Args >
        TaskProperties operator() (Args&&...)
        {
            return TaskProperties{};
        }
    };

    template < typename ImplCallable, typename PropCallable = DefaultPropFunctor >
    auto make_functor( ImplCallable && impl, PropCallable && prop = DefaultPropFunctor{} )
    {
        return TaskFactoryFunctor< ImplCallable, PropCallable >{ *this, impl, prop };
    }
};

} // namespace rmngr

