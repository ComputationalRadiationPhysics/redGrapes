
######################
    Best Practices
######################

.. _best-practice_singleton:

Singleton for Manager
=====================
An easy way to make the manager globally accessible is to create a singleton:

.. code-block:: c++

    using TaskProperties = rg::TaskProperties< /*...*/ >;
    static auto & mgr()
    {
        static rg::Manager<
            TaskProperties,
            rg::ResourceEnqueuePolicy,
            MyScheduler
        > m;

	return m;
    }

    void foo()
    {
        mgr().emplace_task( []{ /* ... */ } );
    }

.. _best-practice_lifetimes:

Lifetimes of Captured Variables
===============================
use ``shared_ptr``


.. _best-practice_task-results:

Task-Results
================================
always use ``auto``


.. _best-practice_containers:

Writing Container Classes
=========================
If you implement a structure which should be used **as** resource, then just derive from the corresponding resource type:

.. code-block:: c++

    struct MyContainer : rg::IOResource {
        // ...
    }

TODO: Access Guards
