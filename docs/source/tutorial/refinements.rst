
######################
    Refining Tasks
######################

It is possible to create a sub-graph inside a task during its execution.
This is done without further thought by just calling ``emplace_task()`` inside another task.
Either you always capture the manager by reference or create a singleton (See :ref:`best-practice_singleton`).

.. code-block:: c++

    mgr.emplace_task(
        [&mgr]
        {
            mgr.emplace_task(
                []{ /* ... */ },
                TaskProperties::Builder().label("Child Task")
            );
        },
        TaskProperties::Builder().label("Parent Task")
    );

Property Constraints
====================

Because the properties of the parent task already made decisions about the scheduling, any child tasks are not allowed to
revert these assumptions. So the properties of child tasks are constrained and assertet at task creation. This is implemented by the :ref:`EnqueuePolicy <concept_EnqueuePolicy>`. In case of using the predefined `ResourceEnqueuePolicy`, it asserts the resource accesses of the parent task to be supersets of its child tasks. That means firstly no new resources should be introduced and secondly all access modes must be less or equally "mutable", e.g. a child task cannot write a resource that is only read by the parent task.

.. note::
   Not meeting the resource constraint will throw an exception when calling ``emplace_task()``. This is only possible because we don't use access guards in this example.

.. code-block:: c++

    rg::Resource< rg::access::IOAccess > r1;

    mgr.emplace_task(
       [&mgr, r1]
       {
           // OK.
           mgr.emplace_task(
               []{ /* ... */ },
               TaskProperties::Builder()
	           .label("good child")
                   .resources({ r1.make_access(rg::access::IOAccess::read) })
	   );

           // throws runtime error
           mgr.emplace_task(
               []{ /* ... */ },
               TaskProperties::Builder()
                   .label("bad child")
                   .resources({ r1.make_access(rg::access::IOAccess::write) })
           );
       },
       TaskProperties::Builder()
           .label("Parent Task")
           .resources({ r1.make_access(rg::access::IOAccess::read) })
   );


Resource Scopes
===============

It is also possible to create resources which exist locally inside a task and are only relevant for sub-tasks.

.. code-block:: c++

    rg::IOResource< int > r1;

    mgr.emplace_task(
        [&mgr]( auto r1 )
        {
            rg::IOResource< int > local_resource;

            mgr.emplace_task(
                []( auto r1, auto r2 ){ /* ... */ },
		TaskProperties::Builder().label("Child Task 1"),
                r1.read(),
                // use local_resource here without violating the subset constraint
                local_resource.write(),
            );

            mgr.emplace_task(
                []( auto r ){ /* ... */ },
                TaskProperties::Builder().label("Child Task 2"),
                local_resource.read()
	    );
	},
	TaskProperties::Builder().label("Parent Task")

        // can't and doesn't need local_resource
        r1.read()
    );


.. note::
   The context in which the constructor of a resource is called determines its *scope-level*.
   Local resources should therefore be constructed inside of the parent task.
