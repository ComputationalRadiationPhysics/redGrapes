
#pragma once

#include <rmngr/thread_dispatcher.hpp>

namespace rmngr
{

template <typename Job>
struct DefaultJobSelector
{
    struct Property {};

    void push( Job const&, Property const& ) {}
    bool empty( void ) { return true; }
    Job getJob( void ) { return Job(); }
};

template <
    template <typename>
    class T_JobSelector = DefaultJobSelector
>
struct DispatchPolicy
  : DefaultSchedulingPolicy
{
    struct RuntimeProperty
    {
        RuntimeProperty() : state(pending) {}

        enum { pending, ready, running, done } state;

        std::string color() const
        {
            switch( state )
            {
                case pending: return "yellow";
                case ready: return "green";
                case running: return "white";
                default: return "gray";
            }
        }
    };

    struct Job
    {
        observer_ptr<SchedulerInterface::SchedulableInterface> schedulable;
        observer_ptr<RuntimeProperty> prop;
        observer_ptr<SchedulerInterface> scheduler;

        void operator() (void)
        {
            if(! schedulable )
              return;

            auto lock = scheduler->lock();
            prop->state = RuntimeProperty::running;
            schedulable->start();

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

    struct ProtoProperty
        : T_JobSelector<Job>::Property
    {};

    struct Worker : public SchedulerInterface::WorkerInterface
    {
        observer_ptr<ThreadDispatcher<JobSelector>> dispatcher;
        void work(void)
        {
            dispatcher->consume_job();
        }
    };

    JobSelector selector;
    ThreadDispatcher<JobSelector> * dispatcher;
    Worker worker;

    DispatchPolicy()
    {
        this->dispatcher = nullptr;
    }

    void init( SchedulerInterface & s )
    {
        this->selector.scheduler = s;
        this->dispatcher =
          new ThreadDispatcher< JobSelector >(
            this->selector,
            s.num_threads()
          );
        this->worker.dispatcher = this->dispatcher;
        s.set_worker( this->worker );
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
                ProtoProperty & proto_prop = *schedulable;
                RuntimeProperty & runtime_prop = *schedulable;

                if ( runtime_prop.state == RuntimeProperty::pending )
                {
                    runtime_prop.state = RuntimeProperty::ready;
                    selector.push( Job{ &schedulable, runtime_prop, scheduler }, proto_prop );
                }
                else if ( runtime_prop.state == RuntimeProperty::done )
                    schedulable->finish();
            }
        }
    }
};

}; // namespace rmngr

