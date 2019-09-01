
#pragma once

#include <rmngr/graph/refined_graph.hpp>
#include <rmngr/graph/precedence_graph.hpp>
#include <rmngr/graph/util.hpp>

#include <rmngr/thread_schedule.hpp>

namespace rmngr
{
    
template <
    typename T_Task,
    typename T_Graph = boost::adjacency_list<
        boost::setS,
        boost::vecS,
        boost::bidirectionalS,
        T_Task*
    >
>
class SchedulingGraph
{
public:
    using P_Graph = T_Graph;
    using Task = T_Task;

    struct Event
    {
    private:
        std::mutex mutex;
        std::condition_variable cv;
        bool state;

    public:
        Event()
            : state( false )
        {}

        auto lock()
        {
            return std::unique_lock< std::mutex >( mutex );
        }

        void notify()
        {
            {
                auto l = lock();
                state = true;
            }
            cv.notify_all();
        }

        void wait()
        {
            auto l = lock();
            cv.wait( l, [this]{ return state; } );
        }
    };

    using EventGraph = boost::adjacency_list<
        boost::listS,
        boost::listS,
        boost::bidirectionalS
    >;
    using EventID = typename boost::graph_traits< EventGraph >::vertex_descriptor;
    using TaskID = typename boost::graph_traits< P_Graph >::vertex_descriptor;

    std::mutex mutex;
    EventGraph m_graph;
    std::unordered_map< EventID, Event > events;
    std::unordered_map< Task* , EventID > after_events;

    struct Job
    {
        Task * task;

        void operator() ()
        {
            (*task)();
        }
    };

    using ThreadSchedule = rmngr::ThreadSchedule< Job >;

    RefinedGraph< T_Graph > & precedence_graph;
    std::vector< ThreadSchedule > schedule;

    EventID null_id;

    SchedulingGraph( RefinedGraph< T_Graph > & precedence_graph, int n_threads )
        : precedence_graph( precedence_graph )
        , schedule( n_threads )
        , finishing( false )
    {
        std::lock_guard< std::mutex > lock( mutex );
        null_id = boost::add_vertex( m_graph );
    }

    std::atomic_bool finishing;
    void finish()
    {
        finishing = true;
    }

    bool empty()
    {
        std::lock_guard< std::mutex > lock( mutex );
        return finishing && (boost::num_vertices( m_graph ) == 1);
    }

    Job make_job( Task * task )
    {
        EventID before_event = make_event();
        EventID after_event = make_event();

        after_events[ task ] = after_event;

        task->hook_before( [this, before_event]{ finish_event( before_event ); events[before_event].wait(); } );
        task->hook_after( [this, after_event]{ finish_event( after_event ); } );

        std::lock_guard< std::mutex > lock( mutex );
        auto task_id = graph_find_vertex( task, precedence_graph.graph() ).first;
        for(
            auto it = boost::out_edges( task_id, precedence_graph.graph() );
            it.first != it.second;
            ++ it.first
        )
            boost::add_edge( (EventID) boost::source( *(it.first), precedence_graph.graph() ), before_event, m_graph );

        auto ref = precedence_graph.find_refinement_containing( task );
        if( ref && ref->parent )
            boost::add_edge( after_event, after_events[ ref->parent ], m_graph );

        return Job{ task };
    }

    void consume_job()
    {
        schedule[ thread::id ].consume( [this]{ return empty(); } );
    }

    void consume_job( std::function<bool()> const & pred )
    {
        schedule[ thread::id ].consume( [this, pred]{ return empty() || pred(); } );
    }

    std::experimental::optional<Task*> get_current_task()
    {
        if( std::experimental::optional<Job> job = schedule[ thread::id ].get_current_job() )
            return std::experimental::optional<Task*>( job->task );
        else
            return std::experimental::nullopt;
    }

    EventID make_event()
    {
        std::lock_guard< std::mutex > lock( mutex );
        EventID event_id = boost::add_vertex( m_graph );
        events.emplace( std::piecewise_construct, std::forward_as_tuple(event_id), std::forward_as_tuple() );

        boost::add_edge( null_id, event_id, m_graph );

        return event_id;
    }

    void notify_event( EventID id )
    {
        std::unique_lock< std::mutex > lock( mutex );
        if( boost::in_degree( id, m_graph ) == 0 )
        {
            events[ id ].notify();

            // collect events to propagate to before to not invalidate the iterators in recursion
            std::vector< EventID > out;
            for( auto it = boost::out_edges( id, m_graph ); it.first != it.second; it.first++ )
                out.push_back( boost::target( *it.first, m_graph ) );

            boost::remove_vertex( id, m_graph );

            lock.unlock();

            // propagate
            for( EventID e : out )
                notify_event( e );

            if( empty() )
            {
                for( auto & thread : schedule )
                    thread.notify();
            }
        }
    }

    void finish_event( EventID id )
    {
        {
            auto cv_lock = events[id].lock();
            std::lock_guard< std::mutex > graph_lock( mutex );
            boost::remove_edge( null_id, id, m_graph );
        }

        notify_event( id );
    }
}; // class SchedulingGraph
    
} // namespace rmngr

