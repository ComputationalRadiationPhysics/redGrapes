
#################
    Debugging
#################

Task Backtraces
===============

Sometimes it is useful to create a backtrace of tasks. This can be done with the manager method ``backtrace()``. It returns a ``std::vector<TaskProperties>``.

.. code-block:: c++

    mgr().emplace_task(
        []
        {
            mgr().emplace_task(
                []
                {
                    int depth = 0;
                    for( auto t : mgr().backtrace() )
                        std::cout << "[" << depth++ << "]" << t.label << std::endl;
                },
                TaskProperties::Builder().label("Child Task")
            );
        },
        TaskProperties::Builder().label("Parent Task")
    );


This will give us the output:

.. code-block::

   [0] Child Task
   [1] Parent Task


Writing out the Task-Graph
==========================
TODO
