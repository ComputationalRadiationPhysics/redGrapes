/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include <rmngr/thread/thread_dispatcher.hpp>
#include <rmngr/graph/scheduling_graph.hpp>
#include <rmngr/task/delayed_functor.hpp>
#include <rmngr/task/working_future.hpp>
#include <rmngr/task/task.hpp>
#include <rmngr/task/task_container.hpp>
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

template< typename TaskProperties >
struct DefaultScheduler
{
    template < typename SchedulingGraph >
    using type = FIFOScheduler< TaskProperties, SchedulingGraph >;
};

template <
    typename TaskProperties = DefaultTaskProperties,
    typename EnqueuePolicy = DefaultEnqueuePolicy< TaskProperties >,
    template <typename> class Scheduler = DefaultScheduler< TaskProperties >::template type
>
class Manager
{
public:
    using TaskID = typename TaskContainer< TaskProperties >::TaskID;
    struct TaskEnqueuePolicy
    {
        TaskContainer< TaskProperties > * task_container;

        bool is_serial(TaskID a, TaskID b)
        {
            return EnqueuePolicy::is_serial(
                       task_container->task_properties(a),
                       task_container->task_properties(b)
                   );
        }

        void assert_superset(TaskID super, TaskID sub)
        {
            EnqueuePolicy::assert_superset(
                task_container->task_properties(super),
                task_container->task_properties(sub)
            );
        }
    };

    using Refinement = QueuedPrecedenceGraph<
        boost::adjacency_list<
            boost::setS,
            boost::listS,
            boost::bidirectionalS,
            TaskID
        >,
        TaskEnqueuePolicy
    >;

    TaskContainer< TaskProperties > task_container;
    Refinement precedence_graph;
    SchedulingGraph< TaskProperties > scheduling_graph;
    ThreadDispatcher< SchedulingGraph< TaskProperties > > thread_dispatcher;
    Scheduler< SchedulingGraph< TaskProperties > > scheduler;

    struct Worker
    {
        SchedulingGraph< TaskProperties > & scheduling_graph;
        void operator() ( std::function<bool()> const & pred )
        {
            while( !pred() )
                scheduling_graph.consume_job( pred );
        }
    };

    Worker worker;

public:
    Manager( int n_threads = std::thread::hardware_concurrency() )
        : precedence_graph( TaskEnqueuePolicy{ &task_container } )
        , scheduling_graph( task_container, precedence_graph, n_threads )
        , scheduler( task_container, scheduling_graph )
        , thread_dispatcher( scheduling_graph, n_threads )
        , worker{ scheduling_graph }
    {}

    ~Manager()
    {
        scheduling_graph.finish();
        thread_dispatcher.finish();
    }

    auto & getScheduler()
    {
        return scheduler;
    }

    template< typename NullaryCallable >
    auto emplace_task( NullaryCallable && impl, TaskProperties const & prop = TaskProperties{} )
    {
        auto delayed = make_delayed_functor( std::move(impl) );
        auto result = make_working_future( std::move(delayed.get_future()), worker );
        this->push( new FunctorTask< TaskProperties, decltype(delayed) >( std::move(delayed), prop ) );
        return result;
    }

    /**
     * Enqueue a Schedulable as child of the current task.
     */
    TaskID push( Task< TaskProperties > * task )
    {
        TaskID id = task_container.emplace( task );
        scheduler.push( id, this->get_current_refinement() );

        return id;
    }

    Refinement &
    get_current_refinement( void )
    {
        if( std::experimental::optional< TaskID > task_id = scheduling_graph.get_current_task() )
        {
            auto r = this->precedence_graph.template refinement<Refinement>( *task_id );

            if(r)
                return *r;
        }

        return this->precedence_graph;
    }

    void update_properties( typename TaskProperties::Patch const & patch )
    {
        if( std::experimental::optional< TaskID > task_id = scheduling_graph.get_current_task() )
            update_properties( *task_id, patch );
        else
            throw std::runtime_error("update_properties: currently no task running");
    }

    void update_properties( TaskID id, typename TaskProperties::Patch const & patch )
    {
        task_container.task_properties(id).apply_patch( patch );
        scheduling_graph.template update_vertex< Refinement >( id );
    }

    std::experimental::optional<std::vector<TaskID>> backtrace()
    {
        if( std::experimental::optional< TaskID > task_id = scheduling_graph.get_current_task() )
            return precedence_graph.backtrace( *task_id );
        else
            return std::experimental::nullopt;
    }

    TaskProperties const & task_properties( TaskID id )
    {
        return task_container.task_properties( id );
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

} // namespace rmngr done
