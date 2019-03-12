
#pragma once

#include <rmngr/thread_dispatcher.hpp>
#include <rmngr/graph/util.hpp>
#include <functional>
namespace rmngr
{

template <typename Job>
struct DefaultJobSelector
{
    struct Property {};

    virtual void update() = 0;

    void finish() {}
    void notify() {}
    bool empty( void ) { return true; }
    void push( Job const&, Property const& ) {}
  //Job getJob( int finished ) { return Job(); }
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

        enum State { pending, ready, running, done } state;

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
        SchedulerInterface::SchedulableInterface * schedulable;
        RuntimeProperty * prop;
        SchedulerInterface * scheduler;

        Job()
            : schedulable(nullptr)
	    , prop(nullptr)
	    , scheduler(nullptr)
        {}

        Job(
            SchedulerInterface::SchedulableInterface * schedulable,
	    RuntimeProperty * prop,
	    SchedulerInterface * scheduler
	)
            : schedulable(schedulable)
	    , prop(prop)
	    , scheduler(scheduler)
        {}

        void operator() (void)
        {
            if(! schedulable)
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
        bool finished;

        JobSelector()
	    : finished(false)
        {}

        bool empty()
        {
            if( finished && T_JobSelector<Job>::empty() && scheduler->empty() )
	    {
	        this->notify();
		return true;
	    }
	    else
	        return false;
        }

        template <typename Pred>
        Job getJob( Pred const & pred )
        {
            return T_JobSelector<Job>::getJob( [&](){ return pred() || this->finished; } );
        }

        void update()
        {
            this->scheduler->update();
        }

        SchedulerInterface * scheduler;
    };

    struct ProtoProperty
        : T_JobSelector<Job>::Property
    {};

    struct Worker
      : public SchedulerInterface::WorkerInterface
    {
        ThreadDispatcher<JobSelector> * dispatcher;
        void work( std::function<bool()> const& pred )
        {
	    while( !pred() )
	        dispatcher->consume_job( pred );
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
        this->selector.scheduler = &s;
        this->dispatcher =
          new ThreadDispatcher< JobSelector >(
            this->selector,
            s.num_threads()
          );
        this->worker.dispatcher = this->dispatcher;
        s.set_worker( this->worker );
    }

    void finish()
    {
        this->selector.finished = true;
        if( this->dispatcher )
            delete this->dispatcher;
    }

    void notify()
    {
        this->selector.notify();
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
                    selector.push( Job( schedulable, &runtime_prop, &scheduler ), proto_prop );
                }
                else if ( runtime_prop.state == RuntimeProperty::done )
                    schedulable->finish();
            }
        }
    }
};

}; // namespace rmngr

