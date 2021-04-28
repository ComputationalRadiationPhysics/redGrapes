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

#include <redGrapes/property/id.hpp>
#include <redGrapes/property/resource.hpp>

#include <redGrapes/scheduler/default_scheduler.hpp>

#include <spdlog/spdlog.h>

#include <sstream>


namespace redGrapes
{

// TODO: find better name, like "root task", "TaskSpace" .. ?
template <
    typename... TaskPropertyPolicies
>
class Manager
{
public:
    struct TaskPtr;
    struct WeakTaskPtr;
    using TaskID = unsigned int;

    using TaskProps = TaskProperties< IDProperty, ResourceProperty, TaskPropertyPolicies...  >;

    struct Task : TaskProps
    {
        std::shared_ptr< TaskImplBase > impl;
        std::experimental::optional< WeakTaskPtr > parent;
        
        template < typename F >
        Task( F && f, TaskProps prop )
            : TaskProps(prop)
            , impl( new FunctorTask< F >( std::move( f ) ) )
        {}
    };

    using PGraph = PrecedenceGraph< Task >;

    struct WeakTaskPtr
    {
        std::weak_ptr< PGraph > graph;
        typename PGraph::VertexID vertex;

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
        std::shared_ptr< PGraph > graph;
        typename PGraph::VertexID vertex;

        Task & get() const
        {
            return graph_get(vertex, graph->graph()).first;
        }

        Task & locked_get() const
        {
            auto lock = graph->shared_lock();
            return graph_get(vertex, graph->graph()).first;
        }

        // TODO: move to PrecedenceGraph
        std::vector< TaskPtr > get_predecessors() const
        {
            std::vector< TaskPtr > predecessors;

            for(
                auto edge_it = boost::in_edges( vertex, graph->graph() );
                edge_it.first != edge_it.second;
                ++edge_it.first
            )
            {
                auto target_vertex =
                    boost::source(
                        *edge_it.first,
                        graph->graph()
                    );

                predecessors.push_back( TaskPtr{ graph, target_vertex } );
            }

            return predecessors;
        }

        std::vector< TaskPtr > get_followers() const
        {
            std::vector< TaskPtr > followers;

            for(
                auto edge_it = boost::out_edges( vertex, graph->graph() );
                edge_it.first != edge_it.second;
                ++edge_it.first
            )
            {
                auto target_vertex =
                    boost::target(
                        *edge_it.first,
                        graph->graph()
                    );

                followers.push_back( TaskPtr{ graph, target_vertex } );
            }

            return followers;
        }
    };

    static std::optional< TaskPtr > & current_task()
    {
        static thread_local std::optional< TaskPtr > current_task;
        return current_task;
    }

    // destruction order is important here! now really?
public:
    std::shared_ptr< SchedulingGraph< TaskID, TaskPtr > > scheduling_graph;
private:
    std::shared_ptr< PGraph > main_graph;
    std::shared_ptr< scheduler::IScheduler< TaskID, TaskPtr > > scheduler;

    template < typename... Args >
    static inline void pass( Args&&... ) {}

    struct PropBuildHelper
    {
        typename TaskProps::Builder & builder;

        template < typename T >
        inline int build( T const & x )
        {
            trait::BuildProperties< T >::build(builder, x);
            return 0;
        }

	void foo() {
	}
    };

public:
    using EventID = typename SchedulingGraph< TaskID, TaskPtr >::EventID;

    Manager(
        std::shared_ptr< PGraph > main_graph
            = std::make_shared< QueuedPrecedenceGraph< Task, ResourceEnqueuePolicy > >()
    )
        : main_graph( main_graph )
        , scheduling_graph(
              std::make_shared< SchedulingGraph< TaskID, TaskPtr > >(
                  [this] ( TaskPtr a, TaskPtr b )
                  {
                      return this->scheduler->task_dependency_type( a, b );
                  }))
    {
        set_scheduler( scheduler::make_default_scheduler( *this ) );
    }

    ~Manager( )
    {
        while( ! scheduling_graph->empty() )
            redGrapes::thread::idle();

        scheduler->notify();
    }

    auto & getScheduler()
    {
        return scheduler;
    }

    /*! Initialize the scheduler to work with this manager.
     * Must be called at initialization before any call to `emplace_task`.
     */
    void set_scheduler( std::shared_ptr< scheduler::IScheduler< TaskID, TaskPtr > > scheduler )
    {
        this->scheduler = scheduler;
        this->scheduler->init_mgr_callbacks(
            scheduling_graph,
            [this] ( TaskPtr task_ptr ) { return run_task( task_ptr ); },
            [this] ( TaskPtr task_ptr ) { activate_followers( task_ptr ); },
            [this] ( TaskPtr task_ptr ) { remove_task( task_ptr ); }
        );
    }

    /*! create a new task, as child of the currently running task (if there is one)
     *
     * @param f callable that takes "proprty-building" objects as args
     * @param args are forwarded to f after the each arg added its
     *             properties to the task
     *
     * For the argument-types can a trait be implemented which
     * defines a hook to add task properties depending the the
     * argument.
     *
     * @return future from f's result
     */
    template <
        typename Callable,
        typename... Args
    >
    auto emplace_task(
        Callable && f,
        Args&&... args
    )
    {
        typename TaskProps::Builder builder;
        return emplace_task( f, builder, std::forward<Args>(args)... );
    }

    /*! create a new task, as child of the currently running task (if there is one)
     *
     * @param f callable that takes "proprty-building" objects as args
     * @param builder used sequentially by property-builders of each arg
     * @param args are forwarded to f after the each arg added its
     *             properties to the task
     *
     * Firstly the task properties get initialized through
     * the builder-object.
     * Secondly, for the argument-types can a trait be implemented which
     * defines a hook to add further task properties depending the the
     * argument.
     *
     * @return future from f's result
     */
    template <
        typename Callable,
        typename... Args
    >
    auto emplace_task(
        Callable && f,
        typename TaskProps::Builder builder,
        Args&&... args
    )
    {
        PropBuildHelper build_helper{ builder };
        pass( build_helper.template build<Args>(args)... );

	build_helper.foo();

        auto impl = std::bind( f, std::forward<Args>(args)... );

        auto delayed = make_delayed_functor( std::move(impl) );
        auto future = delayed.get_future();

        EventID result_event = scheduling_graph->new_event();

        builder.init_id();

        Task task(
            std::bind(
                [this, result_event]( auto && delayed ) mutable
                {
                    delayed();
                    reach_event( result_event );
                },
                std::move(delayed)
            ),
            builder
        );

        spdlog::debug( "emplace_task {}\n", (TaskProps const&)task );

        this->push_task( std::move( task ) );

        return make_working_future( std::move(future), *this, result_event );
    }

    /*! Enqueue a task object into the precedence graph of
     *  the currently running parent task (or the root graph
     *  if there is no parent).
     *
     * @return Wrapper object for accessing task information
     */
    TaskPtr push_task( Task && task )
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
        scheduling_graph->add_task( task_ptr );

        g_lock.unlock();
        {
            auto g_lock = g->shared_lock();
            scheduler->activate_task( task_ptr );
        }

        scheduler->notify();

        return task_ptr;
    }

    /*! Start the execution of a task.
     *
     * @return true if the task finished, false if it was paused.
     */
    bool run_task( TaskPtr task_ptr )
    {
        auto tl = task_ptr.graph->unique_lock(); // use shared_lock ??
        auto impl = task_ptr.get().impl;
        auto task_id = task_ptr.get().task_id;

        spdlog::debug( "run task {}", task_id );

        tl.unlock();

        current_task() = task_ptr;
        bool finished = (*impl)();
        current_task() = std::nullopt;

        return finished;
    }

    //! tell the scheduler to look at all tasks following the given one.
    void activate_followers( TaskPtr task_ptr )
    {
        auto graph_lock = task_ptr.graph->shared_lock();
        spdlog::trace( "activate followers of task {}", task_ptr.get().task_id );

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

            auto p = TaskPtr{ task_ptr.graph, target_vertex };
            scheduler->activate_task( TaskPtr{ task_ptr.graph, target_vertex } );
        }

        graph_lock.unlock();
        scheduler->notify();
    }

    //! remove task from precedence graph and scheduling graph
    void remove_task( TaskPtr task_ptr )
    {
        auto graph_lock = task_ptr.graph->unique_lock();
        spdlog::trace("mgr: remove task {}", task_ptr.get().task_id);
        auto task_id = task_ptr.get().task_id;
        task_ptr.graph->finish( task_ptr.vertex );
        graph_lock.unlock();

        scheduling_graph->remove_task( task_id );
    }

    /*! Get the TaskID of the currently running task.
     * @return nullopt if there is no task running currently.
     */
    std::experimental::optional< TaskID >
    get_current_task_id( )
    {
        if( current_task() )
            return current_task()->locked_get().task_id;
        else
            return std::experimental::nullopt;
    }

    //! flag the state of the event & update
    void reach_event( EventID event_id )
    {
        scheduling_graph->reach_event( event_id );
        scheduler->notify();
    }

    /*! Create an event on which the termination of the current task depends.
     *  A task must currently be running.
     *
     * @return Handle to flag the event with `reach_event` later.
     *         nullopt if there is no task running currently
     */
    std::optional< EventID >
    create_event()
    {
        if( auto task_id = get_current_task_id() )
            return scheduling_graph->add_post_dependency( *task_id );
        else
            return std::nullopt;
    }

    //! get the subgraph which contains all children of the currently running task
    std::shared_ptr< PGraph >
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
                auto new_graph = std::shared_ptr< PGraph >( parent_graph->default_child( parent_graph, task_ptr->vertex ) );
                parent_graph->add_subgraph( task_ptr->vertex, new_graph );
                return new_graph;
            }
            else
                return std::dynamic_pointer_cast< PGraph >( g );
        }
        else
        {
            /* the current thread is not executing a task,
               so we use the root-graph as default */
            return this->main_graph;
        }
    }

    //! apply a patch to the properties of the currently running task
    void update_properties( typename TaskProps::Patch const & patch )
    {
        if( auto task_ptr = current_task() )
        {
            auto lock = task_ptr->graph->unique_lock();
            task_ptr->get().apply_patch( patch );

            auto vertices = task_ptr->graph->update_vertex( task_ptr->vertex );

            std::vector< TaskPtr > followers;
            for( auto v : vertices )
                followers.push_back( TaskPtr{ task_ptr->graph, v } );

            scheduling_graph->update_task( *task_ptr, followers );

            for( auto following_task : followers )
                scheduler->activate_task( following_task );

            lock.unlock();

            scheduler->notify();
        }
        else
            throw std::runtime_error("update_properties: currently no task running");
    }

    /*! wait until all tasks finished
     * can only be called outside of a task
     */
    void wait_for_all()
    {
        spdlog::trace("wait for all tasks...");
        if( ! current_task() )
            while( ! scheduling_graph->empty() )
                thread::idle();
        else
            throw std::runtime_error("called wait_for_all() inside a task!");
    }

    //! pause the currently running task at least until event_id is reached
    void yield( EventID event_id )
    {
        while( ! scheduling_graph->is_event_reached( event_id ) )
        {
            if( current_task() )
            {
                auto & task = current_task()->locked_get();
                spdlog::trace( "pause task {}", task.task_id );
                scheduling_graph->task_pause( task.task_id, event_id );
                task.impl->yield();
            }
            else
                thread::idle();
        }
    }

    //! get backtrace from currently running task
    std::vector< TaskProps >
    backtrace()
    {
        std::vector< TaskProps > bt;

        std::optional< WeakTaskPtr > task_ptr;

        if( auto parent = current_task() )
            task_ptr = WeakTaskPtr( *parent );

        while( task_ptr )
        {
            Task & task = task_ptr->locked_get();
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

