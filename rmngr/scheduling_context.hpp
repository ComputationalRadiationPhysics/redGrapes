
/**
 * @file rmngr/scheduling_context.hpp
 */

#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <fstream>
#include <mutex>
#include <rmngr/functor.hpp>
#include <rmngr/functor_queue.hpp>
#include <rmngr/observer_ptr.hpp>
#include <rmngr/precedence_graph.hpp>
#include <rmngr/resource_user.hpp>
#include <rmngr/scheduler.hpp>
#include <rmngr/scheduling_graph.hpp>
#include <rmngr/thread_dispatcher.hpp>
#include <utility>

namespace rmngr
{

/** Manages scheduling-policies and the transition to dispatching the jobs.
 */
class SchedulingContext
{
  public:
    struct SchedulingInfo : public ResourceUser
    {
        SchedulingInfo( std::vector<ResourceAccess> const & access )
            : ResourceUser( access ), label( "unnamed" ), main_thread( false )
        {
        }
        std::string label;
        bool main_thread;
    };
    /**
     * Base class storing all scheduling info and the functor
     */
    class Schedulable
        : public SchedulingInfo
        , virtual public DelayedFunctorInterface
    {
      public:
        Schedulable( SchedulingInfo const & info )
            : SchedulingInfo( info ), state( pending )
        {
        }

        enum
        {
            pending,
            ready,
            running,
            done,
        } state;
    }; // class Schedulable

    template <typename DelayedFunctor>
    class SchedulableFunctor
        : public DelayedFunctor
        , public Schedulable
    {
      public:
        SchedulableFunctor( DelayedFunctor && f, SchedulingInfo const & info )
            : DelayedFunctor( std::forward<DelayedFunctor>( f ) ),
              Schedulable( info )
        {
        }
    }; // class SchedulableFunctor

    template <typename Functor>
    class ProtoSchedulableFunctor
        : public SchedulingInfo
        , public Functor
    {
      public:
        ProtoSchedulableFunctor(
            Functor const & f,
            SchedulingInfo const & info )
            : SchedulingInfo( info ), Functor( f )
        {
        }

        template <typename DelayedFunctor>
        SchedulableFunctor<DelayedFunctor> *
        clone( DelayedFunctor && f ) const
        {
            return new SchedulableFunctor<DelayedFunctor>(
                std::forward<DelayedFunctor>( f ), *this );
        }
    }; // class ProtoSchedulableFunctor

    struct Executor
    {
        SchedulingContext * context;
        Schedulable * s;

        operator bool() { return s != nullptr; }

        void
        operator()( void )
        {
            context->write_graphviz();
            {
                std::lock_guard<std::mutex> lock( context->graph_mutex );
                s->state = Schedulable::running;
            }
            s->run();
            {
                std::lock_guard<std::mutex> lock( context->graph_mutex );
                s->state = Schedulable::done;
            }
            context->write_graphviz();

            {
                std::lock_guard<std::mutex> lock( context->graph_mutex );
                context->finish( s );
            }
        };
    }; // struct Executor

    struct Scheduler
    {
        SchedulingContext * context;
        FIFO<observer_ptr<Schedulable>> main_fifo, others_fifo;

        void
        push( observer_ptr<Schedulable> s )
        {
            if ( s->main_thread )
                this->main_fifo.push( s );
            else
                this->others_fifo.push( s );
        }

        bool
        empty( void )
        {
            context->update();
            return this->main_fifo.empty() && this->others_fifo.empty();
        }

        Executor
        getJob( void )
        {
            context->update();
            observer_ptr<Schedulable> s( nullptr );
            if ( thread::id == 0 && !main_fifo.empty() )
                s = this->main_fifo.getJob();
            else if ( !others_fifo.empty() )
                s = this->others_fifo.getJob();

            context->current_scheduled[thread::id] = s;
            return Executor{context, s};
        }
    }; // struct Scheduler

    void
    update( void )
    {
        std::lock_guard<std::mutex> lock( this->graph_mutex );
        this->graph.update();

        boost::graph_traits<Graph>::vertex_iterator it, end;
        for ( boost::tie( it, end ) = boost::vertices( this->graph.graph() );
              it != end;
              ++it )
        {
            observer_ptr<Schedulable> s = graph_get( *it, this->graph.graph() );

            if ( this->graph.is_ready( s ) )
            {
                if ( s->state == Schedulable::pending )
                {
                    s->state = Schedulable::ready;
                    this->scheduler.push( s );
                }
                else if ( s->state == Schedulable::done )
                    this->finish( s );
            }
        }
    }

    void
    finish( observer_ptr<Schedulable> s )
    {
        if ( this->graph.finish( s ) )
            delete s;
        this->graph.update();
    }

    using Graph = boost::adjacency_list<
        boost::setS,
        boost::vecS,
        boost::bidirectionalS,
        observer_ptr<Schedulable>>;

    Scheduler scheduler;
    QueuedPrecedenceGraph<Graph, ResourceUser> main_refinement;
    std::mutex graph_mutex;
    SchedulingGraph<Graph> graph;
    ThreadDispatcher<Scheduler> * dispatcher;
    std::vector<observer_ptr<Schedulable>> current_scheduled;

  public:
    SchedulingContext( size_t n_threads = 1 )
        : scheduler{this},
          graph( main_refinement ),
          current_scheduled( n_threads + 1 )
    {
        this->dispatcher =
            new ThreadDispatcher<Scheduler>( this->scheduler, n_threads );
    }

    ~SchedulingContext() { delete this->dispatcher; }

    template <typename Functor>
    ProtoSchedulableFunctor<Functor>
    make_proto( Functor const & f, std::vector<ResourceAccess> access = {} )
    {
        return ProtoSchedulableFunctor<Functor>( f, SchedulingInfo{access} );
    }

    observer_ptr<Schedulable>
    get_current_schedulable( void )
    {
        return this->current_scheduled[thread::id];
    }

    QueuedPrecedenceGraph<Graph, ResourceUser> &
    get_main_refinement( void )
    {
        return this->main_refinement;
    }

    template <typename Refinement>
    observer_ptr<Refinement>
    get_current_refinement( void )
    {
        return this->main_refinement.refinement<Refinement>(
            this->get_current_schedulable() );
    }

    FunctorQueue< QueuedPrecedenceGraph<Graph, ResourceUser> >
    get_main_queue( void )
    {
        return make_functor_queue( this->main_refinement );
    }

    template <typename Refinement>
    FunctorQueue<Refinement>
    get_current_queue( void )
    {
        return make_functor_queue( this->get_current_refinement<Refinement>() );
    }

    void
    write_graphviz( void )
    {
        std::lock_guard<std::mutex> lock( this->graph_mutex );

        static int step = 0;
        ++step;
        std::string name = std::string( "Step " ) + std::to_string( step );
        std::string path = std::string( "step_" ) + std::to_string( step ) +
                           std::string( ".dot" );
        std::cout << "write schedulinggraph to " << path << std::endl;
        std::ofstream file( path );
        this->graph.write_graphviz(
            file,
            boost::make_function_property_map<Schedulable *>(
                []( Schedulable * const & s ) { return s->label; } ),
            boost::make_function_property_map<Schedulable *>(
                []( Schedulable * const & s ) {
                    switch ( s->state )
                    {
                        case Schedulable::done:
                            return std::string( "grey" );
                        case Schedulable::running:
                            return std::string( "green" );
                        case Schedulable::ready:
                            return std::string( "yellow" );
                        default:
                            return std::string( "red" );
                    }
                } ),
            name );

        file.close();
    }
}; // struct SchedulingContext

} // namespace rmngr
