
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
    struct Property;
    struct Job
    {
        SchedulerInterface::TaskInterface * task;
        Property * prop;
        SchedulerInterface * scheduler;

        Job()
            : task(nullptr)
	    , prop(nullptr)
	    , scheduler(nullptr)
        {}

        Job(
            SchedulerInterface::TaskInterface * task,
	    Property * prop,
	    SchedulerInterface * scheduler
	)
            : task(task)
	    , prop(prop)
	    , scheduler(scheduler)
        {}

        void operator() (void)
        {
            if(! task)
                return;

            auto lock = scheduler->lock();
            prop->state = Property::running;

            lock.unlock();
            task->run();
            lock.lock();

            prop->state = Property::done;
            task->finish();
        }
    };

    struct Property
    {
        struct Patch// TODO job selector patch
        {
        };

        Property() : state(pending) {}

        typename T_JobSelector< Job >::Property job_selector_prop;
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

        void apply_patch( Patch const & patch )
        {}
    };

    void update_property( Property & s, typename Property::Patch const & patch )
    {
        s.apply_patch( patch );
    }

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

    struct Worker
      : public SchedulerInterface::WorkerInterface
    {
        ThreadDispatcher<JobSelector> * dispatcher;

        Worker()
            : dispatcher(nullptr) {}

        void work( std::function<bool()> const& pred )
        {
             while( !pred() )
             {
                 if(dispatcher)
		     dispatcher->consume_job( pred );
            }
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
            auto task = graph_get( *(it.first), graph.graph() );
            if ( graph.is_ready( task ) )
            {
                Property & prop = task->template property< DispatchPolicy >();
                if ( prop.state == Property::pending )
                {
                    prop.state = Property::ready;
                    selector.push( Job( task, &prop, &scheduler ), prop.job_selector_prop );
                }
                else if ( prop.state == Property::done )
                    task->finish();
            }
        }
    }
};

}; // namespace rmngr

