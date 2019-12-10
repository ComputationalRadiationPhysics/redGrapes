
###############################
    Asynchronous Operations
###############################

e.g. Compute Kernels, MPI calls need to be represented as tasks, but their execution only consists of starting an asynchronous process. The task however should not finish until the
asynchronous operation is done, but not through blocking inside the task. So we need to delay the removal of the task from the graph. This is done with *events*, which can be registered
inside a task and then can be triggered by some polling loop.

Creating Events
===============

``Manager::create_event()`` creates an event object, on which the current task now depends. That means it will not be removed from the graph before the event is reached, even
if the task itself is done executing.
The removal of the task from the graph can then be triggerd with ``Manager::reach_event( EventID )``. If there are multiple events, the task will disappear when all events are reached.

See `examples/8_event.cpp <https://github.com/ComputationalRadiationPhysics/redGrapes/blob/dev/examples/8_event.cpp>`_

Polling
=======

Instead of blocking, a worker thread can be configured to use a polling function when no tasks are available for this thread.
