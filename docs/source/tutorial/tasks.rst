
#####################
    Task Creation
#####################

To create a task, use the function ``emplace_task()``.
The first parameter is any callable, e.g. function pointer or 'lambda' expression.
``emplace_task()`` is variadic and the remaining arguments are passed to the callable.
After calling ``emplace_task()``, the scheduler is activated automatically to execute the task in one of the worker threads.
``rg::finalize()`` will ensure that all tasks have finished and then proceed to de-initialize redGrapes.

.. code-block:: c++

   #include <iostream>
   #include <redGrapes/redGrapes.hpp>

   int main()
   {
       redGrapes::init();

       redGrapes::emplace_task([] {
           std::cout << "Hello World!" << std::endl;
       });

       redGrapes::finalize();
       return 0;
   }

.. CAUTION::
   Tasks are executed asynchronously, so be sure that all captures outlive the tasks execution.
   For best practice see :ref:`best-practice_lifetimes`.

Return Values
=============

The callable passed to ``emplace_task()`` can have any return type. The result can be retrieved through a future object which is returned by ``emplace_task()``.
Even if no return value is required, calling ``get()`` on such a ``Future<T>`` might be useful for awaiting the finishing of a specific task.

.. code-block:: c++

    auto result = rg::emplace_task( []{ return 123; } );
    assert( result.get() == 123 );

.. CAUTION::
   Always use ``auto`` or ``redGrapes::Future<T>`` on task results. Do not cast them to ``std::future<T>``, deadlocks might occur!
   (See :ref:`best-practice_task-results`)
