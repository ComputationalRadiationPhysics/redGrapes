/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <shared_mutex>
#include <unordered_map>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include <redGrapes/graph/scheduling_graph.hpp>
#include <redGrapes/graph/recursive_graph.hpp>
#include <redGrapes/graph/precedence_graph.hpp>
#include <redGrapes/graph/util.hpp>
#include <redGrapes/task/delayed_functor.hpp>
#include <redGrapes/task/working_future.hpp>
#include <redGrapes/task/task.hpp>

#include <redGrapes/property/inherit.hpp>

#include <redGrapes/property/trait.hpp>

namespace redGrapes
{

template <typename T>
struct DefaultEnqueuePolicy
{
    static bool is_serial(T const & a, T const & b) { return true; }
    static void assert_superset(T const & super, T const & sub) {}
};

template <
    typename T_TaskProperties,// = TaskProperties<>,
    typename EnqueuePolicy,// = DefaultEnqueuePolicy< T_TaskProperties >,
    template <typename, typename, typename> class Scheduler// = FIFOScheduler
>
class Manager
{
public:
    using TaskID = unsigned int;
    static TaskID gen_task_id()
    {
        static TaskID id = 0;
        static std::mutex m;
        std::unique_lock<std::mutex> l(m);
        return id++;
    }

    struct TaskPtr;
    struct WeakTaskPtr;

    struct TaskProperties : T_TaskProperties
    {
        TaskID task_id;
        std::experimental::optional< WeakTaskPtr > parent;

        TaskProperties( T_TaskProperties const & p )
            : T_TaskProperties( p )
            , task_id(gen_task_id())
        {}
    };

    struct Task : TaskProperties
    {
        std::shared_ptr< TaskImplBase > impl;

        template < typename F >
        Task( F && f, T_TaskProperties prop )
            : TaskProperties(prop)
            , impl( new FunctorTask<decltype(std::bind(f, this->task_id))>(std::bind(std::move(f), this->task_id)) )
        {}
    };

    using PrecedenceGraph = QueuedPrecedenceGraph<Task, EnqueuePolicy>;

    struct WeakTaskPtr
    {
        std::weak_ptr< PrecedenceGraph > graph;
        typename PrecedenceGraph::VertexID vertex;

        WeakTaskPtr( TaskPtr const & other )
            : graph( other.graph )
            , vertex( other.vertex )
        {}

        Task & get() const
        {
            return graph_get(vertex, graph.lock()->graph()).first;
        }

        Task & locked_get() const
        {
            auto g = this->graph.lock();
            auto lock = g->shared_lock();
            return graph_get(vertex, g->graph()).first;
        }
    };

    struct TaskPtr
    {
        std::shared_ptr< PrecedenceGraph > graph;
        typename PrecedenceGraph::VertexID vertex;

        Task & get() const
        {
            return graph_get(vertex, graph->graph()).first;
        }

        Task & locked_get() const
        {
            auto lock = graph->shared_lock();
            return graph_get(vertex, graph->graph()).first;
        }
    };

    static std::optional< TaskPtr > & current_task()
    {
        static thread_local std::optional< TaskPtr > current_task;
        return current_task;
    }

    // destruction order is important here!
    SchedulingGraph< TaskID, TaskPtr > scheduling_graph;
    std::shared_ptr< PrecedenceGraph > main_graph;
    Scheduler< TaskID, TaskPtr, PrecedenceGraph > scheduler;

    template < typename... Args >
    static inline void pass( Args&&... ) {}

    struct PropBuildHelper
    {
        typename T_TaskProperties::Builder & builder;

        template < typename T >
        inline int build( T const & x )
        {
            trait::BuildProperties< T >::build(builder, x);
            return 0;
        }
    };

public:
    using EventID = typename Scheduler<TaskID, TaskPtr, PrecedenceGraph>::EventID;

    Manager( )
        : main_graph( std::make_shared<PrecedenceGraph>() )
        , scheduling_graph(
              [this] ( TaskPtr a, TaskPtr b )
              {
                  return this->scheduler.task_dependency_type( a, b );
              })
        , scheduler(
              main_graph,
              scheduling_graph,
              [this] ( TaskPtr task_ptr ) { return run_task( task_ptr ); },
              [this] ( TaskPtr task_ptr ) { finish_task( task_ptr ); }
          )
    {}

    ~Manager()
    {
        while( ! scheduling_graph.empty() )
            redGrapes::thread::idle();

        scheduler.notify();
    }

    auto & getScheduler()
    {
        return scheduler;
    }

    /*! create a new task, as child of the currently running task (if there is one)
     *
     * @param f callable that takes "proprty-building" objects as args
     * @param builder used sequentially by property-builders of each arg
     * @param args are forwarded to f after the each arg added its
     *             properties to the task
     *
     * @return future from f's result
     */
    template <
        typename Callable,
        typename... Args
    >
    auto emplace_task(
        Callable && f,
        typename T_TaskProperties::Builder builder,
        Args&&... args
    )
    {
        PropBuildHelper build_helper{ builder };
        pass( build_helper.template build<Args>(args)... );

        auto impl = std::bind( f, std::forward<Args>(args)... );

        auto delayed = make_delayed_functor( std::move(impl) );
        auto future = delayed.get_future();

        EventID result_event = scheduling_graph.new_event();

        Task task(
            [this, delayed{ std::move(delayed) }, result_event] ( TaskID task_id ) mutable
            {
                scheduling_graph.task_start( task_id );
                delayed();

                reach_event( result_event );
            },
            builder
        );

        this->push( std::move( task ) );

        return make_working_future( std::move(future), *this, result_event );
    }

    template <
        typename Callable,
        typename... Args
    >
    auto emplace_task(
        Callable && f,
        Args&&... args
    )
    {
        typename TaskProperties::Builder builder;
        return emplace_task( f, builder, std::forward<Args>(args)... );
    }

    //! enqueue a child of the current task
    TaskPtr push( Task && task )
    {
        if( auto parent = current_task() )
        {
            task.parent = WeakTaskPtr(*parent);
            task.impl->scope_level = task.parent->locked_get().impl->scope_level + 1;
        }
        else
            task.impl->scope_level = 1;

        auto g = get_current_graph();
        auto g_lock = g->unique_lock();
            auto vertex = g->push( task );
            TaskPtr task_ptr { g, vertex };

            scheduling_graph.add_task( task_ptr );
        g_lock.unlock();

        scheduler.activate_task( task_ptr );

        return task_ptr;
    }

    //! execute task
    bool run_task( TaskPtr task_ptr )
    {
        auto tl = task_ptr.graph->unique_lock();
        auto impl = task_ptr.get().impl;
        auto task_id = task_ptr.get().task_id;
        tl.unlock();

        current_task() = task_ptr;
        std::cout << "run task " << task_id << std::endl;
        bool finished = (*impl)();
        std::cout << "returned/paused task "<< task_id << std::endl;
        current_task() = std::nullopt;

        return finished;
    }

    //! remove task from precedence graph and activate all followers
    void finish_task( TaskPtr task_ptr )
    {
        auto graph_lock = task_ptr.graph->unique_lock();
        auto task_id = task_ptr.get().task_id;

        // collect all following tasks
        std::vector< TaskPtr > followers;
        followers.reserve( boost::out_degree( task_ptr.vertex, task_ptr.graph->graph() ) );

        for(
            auto edge_it = boost::out_edges( task_ptr.vertex, task_ptr.graph->graph() );
            edge_it.first != edge_it.second;
            ++edge_it.first
        )
        {
            auto target_vertex =
                boost::target(
                    *edge_it.first,
                    task_ptr.graph->graph()
                );

            followers.push_back( TaskPtr{ task_ptr.graph, target_vertex } );
        }

        // remove task from graph, so the followers potentially get ready
        task_ptr.graph->finish( task_ptr.vertex );
        graph_lock.unlock();
        scheduling_graph.remove_task( task_id );

        // notify scheduler to consider potentially ready tasks
        for( auto task_ptr : followers )
            scheduler.activate_task( task_ptr );
    }

    std::experimental::optional< TaskID >
    get_current_task_id( void )
    {
        if( current_task() )
            return current_task()->locked_get().task_id;
        else
            return std::experimental::nullopt;
    }

    void reach_event( EventID event_id )
    {
        scheduling_graph.reach_event( event_id );
        scheduler.notify();
    }

    //! create an event on which the termination of the current task depends
    std::optional< EventID >
    create_event()
    {
        if( auto task_id = get_current_task_id() )
            return scheduling_graph.add_post_dependency( *task_id );
        else
            return std::nullopt;
    }

    //! get the subgraph which contains all children of the currently running task
    std::shared_ptr< PrecedenceGraph >
    get_current_graph( void )
    {
        if( auto task_ptr = current_task() )
        {
            auto parent_graph = task_ptr->graph;
            auto l = parent_graph->shared_lock();
            auto g = graph_get( task_ptr->vertex, parent_graph->graph() ).second;
            l.unlock();

            if( !g )
            {
                auto new_graph = std::make_shared< PrecedenceGraph >( parent_graph, task_ptr->vertex );
                parent_graph->add_subgraph( task_ptr->vertex, new_graph );
                return new_graph;
            }
            else
                return std::dynamic_pointer_cast< PrecedenceGraph >( g );
        }
        else
            /* the current thread is not executing a task,
               so we use the root-graph as default */
            return this->main_graph;
    }

    //! apply a patch to the properties of the currently running task
    void update_properties( typename TaskProperties::Patch const & patch )
    {
        if( auto task_ptr = current_task() )
        {
            task_ptr->locked_get().apply_patch( patch );

            auto vertices = scheduling_graph.update_task( task_ptr );
            for( auto v : vertices )
            {
                TaskPtr following_task{ task_ptr.graph, v };
                auto task_id = following_task.locked_get().task_id;

                scheduler.activate_task( task_id );
            }
        }
        else
            throw std::runtime_error("update_properties: currently no task running");
    }

    //! pause the currently running task at least until event_id is reached
    void yield( EventID event_id )
    {
        while( ! scheduling_graph.is_event_reached( event_id ) )
        {
            if( current_task() )
            {
                auto & task = current_task()->locked_get();
                scheduling_graph.task_pause( task.task_id, event_id );
                task.impl->yield();
                std::cout << "resume" << std::endl;
            }
            else
                thread::idle();
        }
    }

    //! get backtrace from currently running task
    std::vector< TaskProperties >
    backtrace()
    {
        std::vector< TaskProperties > bt;

        std::optional< WeakTaskPtr > task_ptr;

        if( auto parent = current_task() )
            task_ptr = WeakTaskPtr( *parent );

        while( task_ptr )
        {
            TaskProperties task = task_ptr->locked_get();
            bt.push_back( task );

            if( task.parent )
                task_ptr = WeakTaskPtr( *task.parent );
            else
                task_ptr = std::experimental::nullopt;
        }

        return bt;
    }
};

} // namespace redGrapes
