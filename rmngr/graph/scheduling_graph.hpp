
#pragma once

#include <mutex>
#include <condition_variable>
#include <boost/graph/adjacency_list.hpp>
#include <rmngr/graph/refined_graph.hpp>
#include <rmngr/graph/precedence_graph.hpp>
#include <rmngr/graph/util.hpp>

#include <rmngr/thread/thread_schedule.hpp>
#include <rmngr/task/task_container.hpp>

namespace rmngr
{
    
template <
    typename TaskProperties,
    typename T_Graph = boost::adjacency_list<
        boost::setS,
        boost::vecS,
        boost::bidirectionalS,
        typename TaskContainer< TaskProperties >::TaskID
    >
>
class SchedulingGraph
{
public:
    using P_Graph = T_Graph;

    using EventGraph = boost::adjacency_list<
        boost::listS,
        boost::listS,
        boost::bidirectionalS
    >;
    using EventID = typename boost::graph_traits< EventGraph >::vertex_descriptor;
    using TaskID = typename P_Graph::vertex_property_type;

    struct Event
    {
    private:
        std::mutex mutex;
        std::condition_variable cv;
        bool state;

    public:
        bool ready;
        TaskID task_id;

        Event()
            : state( false )
            , ready( false )
        {}

        auto lock()
        {
            return std::unique_lock< std::mutex >( mutex );
        }

        void notify()
        {
            if( ready )
            {
                {
                    auto l = lock();
                    state = true;
                }
                cv.notify_all();
            }
        }

        void wait()
        {
            auto l = lock();
            cv.wait( l, [this]{ return state; } );
        }
    };

    struct Job
    {
        TaskContainer< TaskProperties > & tasks;
        TaskID task_id;

        void operator() ()
        {
            tasks.task_run( task_id );
        }
    };

    std::mutex mutex;
    EventGraph m_graph;
    std::unordered_map< EventID, Event > events;
    std::unordered_map< TaskID , EventID > before_events;
    std::unordered_map< TaskID , EventID > after_events;

    using ThreadSchedule = rmngr::ThreadSchedule< Job >;

    TaskContainer< TaskProperties > & tasks;
    RefinedGraph< T_Graph > & precedence_graph;
    std::vector< ThreadSchedule > schedule;
    bool finishing;

    SchedulingGraph( TaskContainer< TaskProperties > & tasks, RefinedGraph< T_Graph > & precedence_graph, int n_threads )
        : precedence_graph( precedence_graph )
        , schedule( n_threads )
        , finishing( false )
        , tasks( tasks )
    {}

    void finish()
    {
        {
            std::lock_guard< std::mutex > lock( mutex );
            finishing = true;
        }
        notify();
    }

    bool empty()
    {
        std::lock_guard< std::mutex > lock( mutex );
        return finishing && (boost::num_vertices( m_graph ) == 0);
    }

    EventID add_post_dependency( TaskID task_id )
    {
        std::lock_guard< std::mutex > lock( mutex );
        EventID id = make_event( task_id );

        boost::add_edge( id, after_events[ task_id ], m_graph );

        return id;
    }

    void add_task( TaskID task_id )
    {
        std::lock_guard< std::mutex > lock( mutex );
        EventID before_event = make_event( task_id );
        EventID after_event = make_event( task_id );

        before_events[ task_id ] = before_event;
        after_events[ task_id ] = after_event;

        auto ref = precedence_graph.find_refinement_containing( task_id );
        if( ref )
        {
            auto l = ref->lock();
            if( auto task_vertex = graph_find_vertex( task_id, ref->graph() ) )
            {
                for(
                    auto it = boost::in_edges( *task_vertex, ref->graph() );
                    it.first != it.second;
                    ++ it.first
                )
                {
                    auto preceding_task_id = graph_get( boost::source( *(it.first), ref->graph() ), ref->graph() );
                    if( after_events.count(preceding_task_id) )
                        boost::add_edge( after_events[ preceding_task_id ], before_event, m_graph );
                }
            }

            if( ref->parent )
            {
                if( after_events.count( *ref->parent ) )
                    boost::add_edge( after_event, after_events[ *ref->parent ], m_graph );
                else
                    throw std::runtime_error("parent post-event doesn't exist!");
            }
        }
        else
            throw std::runtime_error("task not found in precedence graph!");

        tasks.task_hook_before(
            task_id,
            [this, before_event]
            {
                if( !finish_event( before_event ) )
                    events[before_event].wait();
            });

        tasks.task_hook_after(
            task_id,
            [this, after_event]
            {
                if( finish_event( after_event ) )
                    notify();
            });
    }

    template <typename Refinement>
    void update_vertex( TaskID task )
    {
        auto ref = dynamic_cast<Refinement*>(this->precedence_graph.find_refinement_containing( task ));
        std::vector<TaskID> selection = ref->update_vertex( task );

        {
            std::lock_guard< std::mutex > lock( mutex );
            for( TaskID other_task : selection )
                boost::remove_edge( after_events[task], before_events[other_task], m_graph );

            for( TaskID other_task : selection )
                notify_event( before_events[other_task] );
        }

        notify();
    }

    void consume_job( std::function<bool()> const & pred = []{ return false; } )
    {
        schedule[ thread::id ].consume( [this, pred]{ return pred() || empty(); } );
    }

    std::experimental::optional<TaskID> get_current_task()
    {
        if( std::experimental::optional<Job> job = schedule[ thread::id ].get_current_job() )
            return std::experimental::optional<TaskID>( job->task_id );
        else
            return std::experimental::nullopt;
    }

    EventID make_event( TaskID task_id )
    {
        EventID event_id = boost::add_vertex( m_graph );
        events.emplace( std::piecewise_construct, std::forward_as_tuple(event_id), std::forward_as_tuple() );
        events[event_id].task_id = task_id;
        return event_id;
    }

    void notify()
    {
        for( auto & thread : schedule )
            thread.notify();
    }

    void remove_event( EventID id )
    {
        TaskID task_id = events[id].task_id;
        boost::clear_vertex( id, m_graph );
        boost::remove_vertex( id, m_graph );
        events.erase( id );

        if( before_events.count(task_id) && before_events[task_id] == id )
            before_events.erase( task_id );
        if( after_events.count(task_id) && after_events[task_id] == id )
            after_events.erase( task_id );
    }

    bool notify_event( EventID id )
    {
        if( events[id].ready && boost::in_degree( id, m_graph ) == 0 )
        {
            events[ id ].notify();

            // collect events to propagate to before to not invalidate the iterators in recursion
            std::vector< EventID > out;
            for( auto it = boost::out_edges( id, m_graph ); it.first != it.second; it.first++ )
                out.push_back( boost::target( *it.first, m_graph ) );

            remove_event( id );

            // propagate
            for( EventID e : out )
                notify_event( e );

            return true;
        }
        else
            return false;
    }

    bool finish_event( EventID id )
    {
        std::unique_lock< std::mutex > lock( mutex );
        events[ id ].ready = true;

        return notify_event( id );
    }
}; // class SchedulingGraph
    
} // namespace rmngr

