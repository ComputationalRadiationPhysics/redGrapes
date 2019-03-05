
###################################
    Getting Functors Scheduled
###################################

Let's start with a very simple function, which we want to get scheduled.

.. code-block:: c++

    int square_impl( int x ) {
        return x * x;
    }

Configuring the Scheduler
=========================

We have to configure our scheduler with so called scheduling-policies, which add properties to the functors and based on these add edges in the scheduling graph to model functor dependencies.

The most basic scheduling policies, which we will always need, are the :ref:`ResourceUserPolicy <class_ResourceUserPolicy>` and the :ref:`DispatchPolicy <class_DispatchPolicy>`.

The ResourceUserPolicy makes every functor a ResourceUser.
The DispatchPolicy takes the ready functors (i.e. they have no dependencies left) and executes them in a thread-pool. It also needs a :ref:`JobSelector <concept_JobSelector>`, which returns the job that gets executed next. We use a simple FIFO.

We define a Scheduler type with our desired configuration.

.. code-block:: c++

    #include <rmngr/scheduler/scheduler.hpp>
    #include <rmngr/scheduler/resource_user.hpp>
    #include <rmngr/scheduler/dispatch.hpp>
    #include <rmngr/scheduler/fifo.hpp>

    using Scheduler =
        rmngr::Scheduler<
	    boost::mpl::vector<
	       rmngr::ResourceUserPolicy,
	       rmngr::DispatchPolicy< rmngr::FIFO >
	    >
        >;

Either we create an scheduler object in our main, or we use the `rmngr::SchedulerSingleton` instead of `rmngr::Scheduler`. On construction we need to set the number of threads, the thread pool is going to use.

.. code-block:: c++

    {
        Scheduler scheduler( n_threads );
        // ...
    }

If we use the singleton, we need to call `init` and `finish`.

.. code-block:: c++

    using Scheduler =
        rmngr::SchedulerSigleton<
            boost::mpl::vector< ... >
        >;

    {
        Scheduler::init( n_threads );
	// ...
	Scheduler::finish();
    }

Creating and Enqueueing Schedulables
====================================

