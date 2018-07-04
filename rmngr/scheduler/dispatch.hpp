
#pragma once

#include <rmngr/thread_dispatcher.hpp>

namespace rmngr
{

template <
    template <typename>
    class T_JobSelector
>
struct DispatchPolicy : DefaultSchedulingPolicy
{
    struct RuntimeProperty
    {
        RuntimeProperty() : state(pending) {}

        enum { pending, ready, running, done } state;
    };

    struct Job
    {
        observer_ptr<SchedulerInterface::SchedulableInterface> schedulable;
        observer_ptr<RuntimeProperty> prop;
        observer_ptr<SchedulerInterface> scheduler;

        operator bool() const
        {
            return bool(prop);
        }

        void operator() (void)
        {
            auto lock = scheduler->lock();
            prop->state = RuntimeProperty::running;

            lock.unlock();
            schedulable->run();
            lock.lock();

            prop->state = RuntimeProperty::done;
            schedulable->finish();
        }
    };

    struct JobSelector : T_JobSelector< Job >
    {
        bool empty()
        {
            return T_JobSelector<Job>::empty() && scheduler->empty();
        }

        observer_ptr<SchedulerInterface> scheduler;
    };

    JobSelector selector;
    ThreadDispatcher<JobSelector> * dispatcher;

    DispatchPolicy()
    {
        this->dispatcher = nullptr;
    }

    void init( observer_ptr<SchedulerInterface> s, int nthreads = 1 )
    {
        this->selector.scheduler = s;
        this->dispatcher =
          new ThreadDispatcher< JobSelector >(
            this->selector,
            nthreads
          );
    }

    ~DispatchPolicy()
    {
        if( this->dispatcher )
            delete this->dispatcher;
    }

    template <typename Graph>
    void update( Graph & graph, SchedulerInterface & scheduler )
    {
        for(
            auto it = boost::vertices( graph.graph() );
            it.first != it.second;
            ++it.first
        )
        {
            auto schedulable = graph_get( *(it.first), graph.graph() );
            if ( graph.is_ready( schedulable ) )
            {
                RuntimeProperty & prop = schedulable;

                if ( prop.state == RuntimeProperty::pending )
                {
                    prop.state = RuntimeProperty::ready;
                    selector.push( Job{ &schedulable, prop, scheduler } );
                }
                else if ( prop.state == RuntimeProperty::done )
                    schedulable->finish();
            }
        }
    }
};

}; // namespace rmngr

