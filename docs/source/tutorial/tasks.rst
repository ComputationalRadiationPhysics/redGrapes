
#####################
    Task Creation
#####################

The Manager
===========

The very first thing to do in every application using RedGrapes is to create a *manager*.
It combines all required components and provides us with an interface for creating tasks.
You also might want to create a namespace alias.

.. code-block:: c++

   #include <redGrapes/manager.hpp>

   namespace rg = redGrapes;

   int main()
   {
       rg::Manager<> mgr;

       return 0;
   }

Its template arguments allow an application specific configuration and are discussed in the following sections (see also :ref:`extend_task-properties` as well as :ref:`custom_schedulers`), but it is also usable with defaults.
The runtime parameter is the number of worker threads which are created additionally to the main thread. By default, it uses the result of ``std::hardware_concurrency()``.

.. code-block:: c++

    rg::Manager<TaskProperties, EnqueuePolicy, Scheduler> mgr( n_threads );

By the the manager-object's destructor, the thread (which is the main thread) will behave as additional worker thread until all
tasks are consumed. Only then the destruction of the manager returns.

Starting a Task
===============

To create a task, the manager method ``emplace_task()`` is used. The first parameter is any nullary callable.
By using ``emplace_task()`` the scheduler is automatically activated and the task will get scheduled and executed in one of the worker threads.

.. code-block:: c++

   #include <iostream>
   #include <redGrapes/manager.hpp>

   namespace rg = redGrapes;

   int main()
   {
       rg::Manager<> mgr;

       mgr.emplace_task(
           []
	   {
	       std::cout << "Hello World!" << std::endl;
	   }
       );

       return 0;
   }

.. CAUTION::
   Tasks are executed asynchronously, so be sure that all captures outlive the tasks execution.
   For best practice see :ref:`best-practice_lifetimes`.

Return Values
=============

The callable passed to ``emplace_task()`` can have any return type. The result can be retrieved through a future object which is returned by ``emplace_task()``.

.. code-block:: c++

    auto result = mgr.emplace_task( []{ return 123; } );
    assert( result.get() == 123 );

.. CAUTION::
   Always use ``auto`` on task results. Do not cast them to ``std::future``, deadlocks might occur!
   (See :ref:`best-practice_task-results`)
