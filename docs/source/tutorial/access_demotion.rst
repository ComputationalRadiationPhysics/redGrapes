
#######################
    Access Demotion
#######################

A very cool feature is that functors can modify their own properties while runnig.
This allows us for example to demote resource accesses so that other functors can start earlier.
Of course the possible changes at runtime have to be well constrained, similarly to :ref:`creating sub-tasks <tutorial_property_constraints>`.

This is done by creating a **patch** in the same manner with builders as the initial properties. This patch is then applied to the current task by the manager method ``update_properties()``. This method must be called inside of a task and applies for exactly the task it is called in.
This call also automatically triggers the scheduler to reevaluate outgoing edges in the task-graph.

The builder :ref:`ResourceProperty <class_ResourceProperty>` provides in its builder interface the methods ``add_resources()`` and ``remove_resources`` for changing the resource access information.

.. note::
    In the case of :ref:`ResourceProperty <class_ResourceProperty>` it is only possible to **demote** the access, i.e. the new access has to be a subset of the old (e.g. we can change a write to read).

.. caution::
   When using access demotion, it is possible again to mess up the actual resource usage and properties, despite access guards, because we can't "delete" a symbol inside a scope.

.. code-block:: c++

    rg::IOResource< int > r1;

    rg::emplace_task(
        []( auto r1 )
        {
            // OK.
            rg::update_properties(
                TaskProperties::Patch::Builder()
                    .remove_resources({ r1.write() })
                    .add_resources({ r1.read() })
            );

	    // compiles, but is wrong
	    // be sure to avoid this
	    *r1 = 123;

	    // throws runtime error, only demotion allowed
            rg::update_properties(
                TaskProperties::Patch::Builder()
                    .add_resources({ r1.write() })
            );
        },
        r1.write()
    );
