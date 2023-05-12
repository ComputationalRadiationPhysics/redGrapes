
############################
    Describing Dataflows
############################

Dataflows occour whenever tasks share any kind of data, i.e. one task outputs data which is used as input for the next. Dataflows between tasks determine their dependencies, i.e. which tasks must be absolutely kept in order and serial.
In RedGrapes this is expressed using *resources*. Each resource represents shared data. Their possible usage by tasks is modelled by an *access policy*, which defines all possible access modes for a task on this resource, e.g. *read*/*write*. An specific configuration of a resource and its access mode is called *resource access*. Tasks can now store a list of resoruce accesses in their properties which is then used to derive the task precedence.

Task Dependencies
=================
When creating a new task, it is inserted into the precedence graph based on an *EqueuePolicy*, which compares the properties of two tasks and decides whether they are dependent. This is done in reverse with all previously inserted tasks to calculate the task dependencies. The manager must be configured with an enqueue policy. ``redGrapes::ResourceEnqueuePolicy`` is predefined and uses the resource properties which are defined
with ``redGrapes::ResourceProperty``.

Resources
=========

The next thing to do is to represent the resources in your code. Any data that is shared between tasks should be represented as resource. Generally resources are just identifiers but there are also wrappers which are memory managed to make resource usage more safe.
A very simple, predefined access policy is ``IOAccess``. It supports the access modes *read* and *write*, where reads can be executed independently.

.. code-block:: c++

    #include <redGrapes/resource/resource.hpp>
    #include <redGrapes/access/io.hpp>

    // just an identifier, no association with actual data
    rg::Resource< rg::access::IOAccess > r1;

Resource Access
---------------
Resource accesses are created with the method ``Resource::make_access(AccessPolicy)`` and can be added to tasks like normal properties. This is the information used by the enqueue policy.

.. code-block:: c++

    rg::emplace_task(
        []{ /* ... */ }
    ).resources({ r1.make_access( rg::access::IOAccess::read ) });

Shared Resource Objects
-----------------------
Using just the previously described mechanisms would require for each shared object an additional resource object and doesn't give any guarantees about what is actually done in the task.
So we could just get the resource accesses wrong and don't know about it. Furthermore the data must absolutely outlive the execution of all tasks.

``rg::SharedResourceObject< T, AccessPolicy >`` is an ``Resource<AccessPolicy>`` and additionally stores an ``shared_ptr<T>``. So we firstly have the data and the resource identifier
united into one object and secondly all lifetime issues are solved through reference counting.

.. tip::
   To avoid lifetime issues, be strict and never capture anything by reference. Only allow copy and move captures.

Access Guards
-------------
By manually adding the resource accesses to the task properties we still cannot check if all operations inside the task are correctly represented by them. The solution to this problem
are *access guards*: Wrappers around a *shared resource object*, for each possible access mode one, that only allows the operations corresponding to the access. For *read*/*write* this
would be an dereference to ``T const&`` or ``T&`` respectively.

Additionally we need to create both the guard object and the task property together with one expression. This is done with so called
*property building parameters*. These are function parameters which are bound to the task immediately at creation (to make it ultimately nullary again), but additionally implement a trait in which they can use the property-builder to modify the task properties. Each access-guard simply implements this trait and so by taking all resources by parameter instead of capture we can use the correct wrapper.

See also :ref:`new_resource_types`.

For convenience the guard objects also provide methods to create new guard objects with a subset of the access.

.. code-block:: c++

    #include <redGrapes/resource/ioresource.hpp>

    rg::IOResource< int > r1;

    rg::emplace_task(
        []( auto r1 )
        {
	    // ok.
            std::cout << *r1 << std::endl;

            // compile-time error!
            *r1 = 123;
        },
	r1.read()
    );

.. tip::
   Altough it is possible to capture resources and add their properties via builders, it is recommended to access them through the parameters, because then the resource usage in the task is checked at compile time.


Full Example
============

In this example `Task 2` and `Task 3` will be executed after `Task 1`. When enough threads are available, `Task 2` and `Task 3` will run in parallel.

.. code-block:: c++

   #include <redGrapes/redGrapes.hpp>
   #include <redGrapes/resource/ioresource.hpp>

   namespace rg = redGrapes;

   int main()
   {
       rg::init();
       rg::IOResource< int > a;

       rg::emplace_task(
           []( auto a ){ *a = 123; },
           a.write()
       ).label("Task 1");

       rg::emplace_task(
           []( auto a ){ int x = *a; },
           a.read()
       ).label("Task 2");

       rg::emplace_task(
           []( auto a ){ int x = *a; },
           a.read()
       ).label("Task 3");

       return 0;
   }
