
###################################
    Getting Functors Scheduled
###################################

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

Let's start with a very simple function, which we want to get scheduled.

.. code-block:: c++

    int square_impl( int x ) {
        return x * x;
    }

Because functors may have arguments, they get wrapped to so called proto-functors, which associate them with their proto-properties (among other properties the resource-usage).
Then these get wrapped to delaying-functors, i.e. when called (`operator() (args...)`), they bind the args and push the new nullary functor to a queue.

So we now wrap our function implementation accordingly:

.. code-block:: c++

    auto square_proto = scheduler.make_proto( &square_impl );
    auto square = scheduler.make_functor( &square_proto );

We can now call `square` like `square_impl`, but the return type is now a future, because the implementation is not executed immediately, rather the functor gets sorted in the scheduling graph for eventual consumation from the thread pool.

.. code-block:: c++

    auto result = square( 4 );
    result.get() == 16;

.. CAUTION::
   Be sure that you use `auto`, because the returned future is wrapped so it can execute jobs instead of waiting. Downcasting to `std::future` will cause deadlocks.

Annotating Properties (Resource Usage)
======================================

Now it is also possible to set properties. We get the property structs defined through the scheduling policies with `.proto_property< SchedulingPolicy >()`.

For `ResourceUserPolicy` this is `ResourceUser`. So we can set the resource access of a functor:

.. code-block:: c++

    ResourceUser& resource_user = square_proto.proto_property< rmngr::ResourceUserPolicy >();
    resource_user.access_list = { a.read(), b.write() };


Full Example
============

With all this we can now build a full example which defines functors with resource acesses and schedules them accordingly.

.. code-block:: c++

    #include <iostream>
    #include <thread>
    #include <rmngr/scheduler/scheduler.hpp>
    #include <rmngr/scheduler/resource_user.hpp>
    #include <rmngr/scheduler/dispatch.hpp>
    #include <rmngr/scheduler/fifo.hpp>
    #include <rmngr/resource/ioresource.hpp>

    int main() {
        rmngr::Scheduler<
            boost::mpl::vector<
                rmngr::ResourceUserPolicy,
                rmngr::DispatchPolicy< rmngr::FIFO >
            >
        > scheduler(
	    std::thread::hardware_concurrency()
	);

	IOResource res;

	auto fun1_proto = scheduler.make_proto(
	    []() { std::cout << "Read" << std::endl; }
	);
	fun1_proto.proto_property< rmngr::ResourceUserPolicy >()
	    .access_list = { res.read() };
        auto fun1 = scheduler.make_functor( fun1_proto );

        auto fun2_proto = scheduler.make_proto(
            []() { std::cout << "Write" << std::endl; }
        );
        fun2_proto.proto_property< rmngr::ResourceUserPolicy >()
            .access_list = { res.write() };
        auto fun2 = scheduler.make_functor( fun2_proto );

        for(int i = 0; i < 10; ++i) {
            fun1();
	    fun1();
	    fun2();
        }

        return 0;
    }
