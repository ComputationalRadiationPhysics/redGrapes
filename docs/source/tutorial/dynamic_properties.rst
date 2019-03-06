
##########################
    Dynamic Properties
##########################

Imagine you want to write a function, which operates on a resource given as argument.

.. code-block:: c++

    struct MyContainer : Resource<MyAccess> { /* ... */ }
    void fun_impl( MyContainer& x ) {
        // do something with x
    }

In this case you can't specify the resource usage statically for the proto-functor, like we did before.

.. code-block:: c++

    auto fun_proto = scheduler.make_proto( &fun_impl );
    fun_proto.proto_property< rmngr::ResourceUserPolicy >().access_list = {
        /* Resource usage unknown here */
    }

    auto fun = scheduler.make_functor( fun_proto );

    MyContainer a, b;
    fun( a ); // resource usage known here
    fun( b ); // and potentially different for every call

Thats where "preparing proto functors" come in.
What these do is knowing the arguments the functor gets called with and using them to execute a "property functor" after cloning (i.e. before pushing it to the queue).
This allows us to set all kinds of properties for the functor while knowing the arguments it gets called with.

.. code-block:: c++

    void fun_prop( Scheduler::SchedulablePtr s, MyContainer& x ) {
        s->proto_property< rmngr::ResourceUserProperty >()
	    .access_list.push_back( x.read() );
    }
    auto fun_proto = scheduler.make_proto( &fun_impl, &fun_prop );
    auto fun = scheduler.make_functor( fun_proto );

    MyContainer a, b;
    fun( a ); // access_list = { a.read() }
    fun( b ); // access_list = { b.read() }
