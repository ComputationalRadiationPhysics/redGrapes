
#pragma once

#include <mutex>
#include <condition_variable>
#include <boost/graph/adjacency_list.hpp>
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
        bool ready;

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
    std::unordered_map< Task* , EventID > before_events;
    std::unordered_map< Task* , EventID > after_events;
    std::unordered_map< EventID, Task* > event_tasks;

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
    }

    bool finishing;
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
        //std::cout << "empty? events remaining: " << boost::num_vertices( m_graph ) << std::endl;
        return finishing && (boost::num_vertices( m_graph ) == 0);
    }

    Job make_job( Task * task )
    {
        std::lock_guard< std::mutex > lock( mutex );
        EventID before_event = make_event();
        EventID after_event = make_event();

        before_events[ task ] = before_event;
        after_events[ task ] = after_event;
        event_tasks[ before_event ] = task;
        event_tasks[ after_event ] = task;

        auto ref = precedence_graph.find_refinement_containing( task );
        if( ref )
        {
            auto l = ref->lock();
            if( auto task_id = graph_find_vertex( task, ref->graph() ) )
            {
                for(
                    auto it = boost::in_edges( *task_id, ref->graph() );
                    it.first != it.second;
                    ++ it.first
                )
                {
                    auto v_id = boost::source( *(it.first), ref->graph() );
                    auto preceding_task = graph_get( v_id, ref->graph() );
                    if( after_events.count(preceding_task) )
                        boost::add_edge( after_events[ preceding_task ], before_event, m_graph );
                }
            }

            if( ref->parent )
            {
                if( after_events.count( ref->parent ) )
                    boost::add_edge( after_event, after_events[ ref->parent ], m_graph );
                else
                    throw std::runtime_error("parent post-event doesn't exist!");
            }
        }
        else
            throw std::runtime_error("task not found in precedence graph!");

        task->hook_before( [this, before_event, task]
            {
                if( !finish_event( before_event ) )
                    events[before_event].wait();
            });
        task->hook_after( [this, after_event]
            {
                if( finish_event( after_event ) )
                    notify();
            });

        return Job{ task };
    }

    template <typename Refinement>
    void update_vertex( Task * task )
    {
        auto ref = dynamic_cast<Refinement*>(this->precedence_graph.find_refinement_containing( task ));
        std::vector<Task*> selection = ref->update_vertex( task );

        {
            std::lock_guard< std::mutex > lock( mutex );
            for( Task * other_task : selection )
                boost::remove_edge( after_events[task], before_events[other_task], m_graph );

            for( Task * other_task : selection )
                notify_event( before_events[other_task] );
        }

        notify();
    }

    void consume_job( std::function<bool()> const & pred = []{ return false; } )
    {
        schedule[ thread::id ].consume( [this, pred]{ return pred() || empty(); } );
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
        EventID event_id = boost::add_vertex( m_graph );
        events.emplace( std::piecewise_construct, std::forward_as_tuple(event_id), std::forward_as_tuple() );
        return event_id;
    }

    void notify()
    {
        for( auto & thread : schedule )
            thread.notify();
    }

    void remove_event( EventID id )
    {
        boost::clear_vertex( id, m_graph );
        boost::remove_vertex( id, m_graph );
        events.erase( id );

        Task * task = event_tasks[id];
        if( before_events.count(task) && before_events[task] == id )
            before_events.erase( task );
        if( after_events.count(task) && after_events[task] == id )
            after_events.erase(task);

        event_tasks.erase( id );
    }

    bool notify_event( EventID id )
    {
        //std::cout << "notify event " << id << std::endl;
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

