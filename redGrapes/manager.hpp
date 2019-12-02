/* Copyright 2019 Michael Sippel
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

#include <redGrapes/thread/thread_dispatcher.hpp>
#include <redGrapes/graph/scheduling_graph.hpp>
#include <redGrapes/task/delayed_functor.hpp>
#include <redGrapes/task/working_future.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/graph/recursive_graph.hpp>
#include <redGrapes/graph/precedence_graph.hpp>
#include <redGrapes/graph/util.hpp>

#include <redGrapes/scheduler/fifo.hpp>

namespace redGrapes
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
    template <typename, typename, typename> class Scheduler = FIFOScheduler
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
        //        std::cout << "task id = " << id << std::endl;
        return id++;
    }

    struct Task : TaskProperties
    {
        std::shared_ptr< TaskImplBase > impl;
        TaskID task_id;
        std::experimental::optional<TaskID> parent_id;

        template <typename F>
        Task( F && f, TaskProperties prop )
            : TaskProperties(prop)
            , impl( new FunctorTask<F>(std::move(f)) )
            , task_id(gen_task_id())
        {}

        void hook_before( std::function<void()> hook )
        {
            impl->before_hooks.push_back( hook );
        }

        void hook_after( std::function<void()> hook )
        {
            impl->after_hooks.push_back( hook );
        }
    };

    using PrecedenceGraph = QueuedPrecedenceGraph<Task, EnqueuePolicy>;

    struct TaskPtr
    {
        std::shared_ptr< PrecedenceGraph > graph;
        typename PrecedenceGraph::VertexID vertex;

        Task & get() const
        {
            return graph_get(vertex, graph->graph()).first;
        }
    };

    std::shared_mutex tasks_mutex;
    std::unordered_map< TaskID, TaskPtr > tasks;

    std::shared_ptr<PrecedenceGraph> main_graph;
    Scheduler< TaskID, TaskPtr, PrecedenceGraph > scheduler;
    ThreadDispatcher< Scheduler<TaskID, TaskPtr, PrecedenceGraph> > thread_dispatcher;

public:
    Manager( int n_threads = std::thread::hardware_concurrency() )
        : main_graph( std::make_shared<PrecedenceGraph>() )
        , scheduler( main_graph, n_threads )
        , thread_dispatcher( scheduler, n_threads )
    {}

    ~Manager()
    {
        scheduler.finish();
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
        auto result = make_working_future( std::move(delayed.get_future()), scheduler );
        this->push( Task(std::move(delayed), prop ) );
        return result;
    }

    /**
     * Enqueue a Schedulable as child of the current task.
     */
    void push( Task && task )
    {
        task.parent_id = get_current_task();

        unsigned int scope_level = thread::scope_level + 1;
        task.hook_before([scope_level]{ thread::scope_level = scope_level; });

        task.hook_after([this, task_id=task.task_id]
        {
            std::unique_lock<std::shared_mutex> lock( tasks_mutex );
            this->tasks.erase( task_id );
        });


        auto task_ptr = scheduler.add_task( task, this->get_current_graph() );

        std::unique_lock<std::shared_mutex> lock( tasks_mutex );
        this->tasks[ task.task_id ] = task_ptr;

    }

    std::experimental::optional<TaskID> get_current_task( void )
    {
        if( auto task_ptr = scheduler.get_current_task() )
        {
            auto l = task_ptr->graph->shared_lock();
            return task_ptr->get().task_id;
        }
        else
            return std::experimental::nullopt;
    }

    std::shared_ptr<PrecedenceGraph>
    get_current_graph( void )
    {
        if( auto task_ptr = scheduler.get_current_task() )
        {
            auto l = task_ptr->graph->shared_lock();
            auto g = graph_get(task_ptr->vertex, task_ptr->graph->graph()).second;
            l.unlock();

            if( !g )
            {
                auto new_graph = std::make_shared<PrecedenceGraph>( task_ptr->graph, task_ptr->vertex );
                task_ptr->graph->add_subgraph( task_ptr->vertex, new_graph );
                return new_graph;
            }
            else
                return std::dynamic_pointer_cast<PrecedenceGraph>(g);
        }

        return this->main_graph;
    }

    void update_properties( typename TaskProperties::Patch const & patch )
    {
        if( auto task_ptr = scheduler.get_current_task() )
        {
            auto l = task_ptr->graph->shared_lock();
            task_ptr->get().apply_patch( patch );
            scheduler.update_vertex( *task_ptr );
        }
        else
            throw std::runtime_error("update_properties: currently no task running");
    }

    std::vector<TaskID> backtrace()
    {
        std::vector<TaskID> bt;
        /*
        auto task_id = get_current_task();
        while( task_id )
        {
            bt.push_back( *task_id );

            auto l = task_ptr.graph->shared_lock();
            task_ptr = tasks[task_id].get().parent_id;
        }
        */
        return bt;
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

} // namespace redGrapes
