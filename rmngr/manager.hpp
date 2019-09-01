
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

namespace rmngr
{

template < typename SchedulingGraph >
struct FIFOScheduler
{
    SchedulingGraph & graph;

    using Task = typename SchedulingGraph::Task;
    using TaskID = typename boost::graph_traits< typename SchedulingGraph::P_Graph >::vertex_descriptor;

    enum State { pending = 0, ready, running, done };
    std::unordered_map< Task*, State > states;

    FIFOScheduler( SchedulingGraph & graph )
        : graph(graph)
    {}

    void notify()
    {
        for( auto & thread : graph.schedule )
        {
            if( thread.empty() )
                schedule( thread );
            else if( auto j = thread.get_current_job() )
                if( states[(*j).task] == done )
                    schedule( thread );
        }
    }

    std::experimental::optional<Task*> find_task( std::function<bool(Task*)> pred )
    {
        for(
            auto it = boost::vertices(graph.precedence_graph.graph());
            it.first != it.second;
            ++ it.first
        )
        {
            auto task_id = *(it.first);
            auto task = graph_get( task_id, graph.precedence_graph.graph() );
            if( pred( task ) )
                return std::experimental::optional<Task*>(task);
        }

        return std::experimental::nullopt;
    }

    bool is_task_ready( Task * task )
    {
        auto task_id = graph_find_vertex( task, graph.precedence_graph.graph()  );
        return boost::out_degree( task_id.first, graph.precedence_graph.graph() ) == 0;
    }

    void make_task_ready( Task * task )
    {
        task->hook_before( [this, task] { states[ task ] = running; } );
        task->hook_after( [this, task] { states[ task ] = done; } );

        states[ task ] = ready;
    }

    void remove_done_tasks()
    {
        for(
            auto it = boost::vertices(graph.precedence_graph.graph());
            it.first != it.second;
            ++ it.first
        )
        {
            auto task = graph_get( *(it.first), graph.precedence_graph.graph() );
            if( states[ task ] == done )
                graph.precedence_graph.finish( task );
        }
    }

    void schedule( typename SchedulingGraph::ThreadSchedule & thread )
    {
        std::unique_lock<std::mutex> lock(graph.mutex);

        remove_done_tasks();

        if( std::experimental::optional<Task *> task = find_task(
                [this]( Task * task )
                {
                    return
                        states[ task ] == pending &&
                        is_task_ready( task );
                } ))
        {
            make_task_ready( *task );
            (*task)->hook_after( [this, &thread]{ notify(); } );

            lock.unlock();
            thread.push( graph.make_job( *task ) );
        }
    }
};

struct DefaultTaskProperties
{
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
    template <typename> typename Scheduler = FIFOScheduler
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

        void work( std::function<bool()> const& pred )
        {
             while( !pred() )
                 scheduling_graph.consume( pred );
        }
    };

    Refinement precedence_graph;

    SchedulingGraph< Task > scheduling_graph;
    Scheduler< SchedulingGraph<Task> > scheduler;
    ThreadDispatcher< SchedulingGraph<Task> > thread_dispatcher;
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
        auto result = make_working_future( delayed.get_future(), this->worker );
        this->push( new FunctorTask< decltype(delayed) >( std::move(delayed), prop ) );
        return result;
    }

    /**
     * Enqueue a Schedulable as child of the current task.
     */
    void push( Task * task )
    {
        this->get_current_refinement().push( task );
        this->scheduler.notify();
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

};

} // namespace rmngr

