
######################
    Refining Tasks
######################

It is possible to create a sub-graph of functors inside a functor during execution as well.
If you use `scheduler.make_functor()` (which you shoud), then you don't have to care for anything.
Just call functors inside your functor implementation.

.. code-block:: c++

    auto fun1 = scheduler.make_functor(scheduler.make_proto(
        [&]( int x ) { /* ... */ }
    ));
    auto fun2 = scheduler.make_functor(scheduler.make_proto(
        [&]( int x ) {
	    // ...
            fun1(x);
	    // ...
	}
    ));


However if you use :ref:`functor queues <class_FunctorQueue>` explicitly, then you should get the right queue inside your functor:

.. code-block:: c++

    auto fun1_proto = scheduler.make_proto( [&](){/* ... */} );
    auto fun2 = main_queue.make_functor( scheduler.make_proto(
        [&]( int x ) {
	    auto queue = scheduler.get_current_queue();
	    auto fun1 = queue.make_functor( fun1_proto );

	    fun1(x);
	}
    ));
