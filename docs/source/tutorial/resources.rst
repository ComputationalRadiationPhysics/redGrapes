
############################
    Describing Dataflows
############################

Dataflows occour whenever tasks use shared data, which is here modelled as *resources*.

Task Dependencies
=================

Enqueue Policy

.. code-block:: c++

    rg::Manager<
        TaskProperties,
        rg::ResourceEnqueuePolicy
    > mgr;


Resources
=========

The next thing to do is to represent the resources in your code. Any data that is shared between tasks should be represented as resource. A very simple, predefined resource-type is ``rg::IOResource``. It supports the access modes *read*, *write*, *atomic add* and *atomic mul*.

.. code-block:: c++

    #include <redGrapes/resource/ioresource.hpp>

    rg::IOResource a;

Resources are copyable:

.. code-block:: c++

    rg::IOResource b( a ); // b identifies the same resource as a

Resource Access
---------------

``rg::IOResource`` also defines convenience wrappers for its access modes, so for example you can call ``read()`` to create a *ResourceAccess* which represents a read-acces on the resource.

.. code-block:: c++

    auto access1 = a.read();
    auto access2 = a.write();

Resource Property
-----------------

.. code-block:: c++

    TaskProperties::Builder().resources({ access1, access2 });


Full Example
============

In this example `Task 2` and `Task 3` will be executed after `Task 1`. When enough threads are available, `Task 2` and `Task 3` will run in parallel.

.. code-block:: c++

   #include <redGrapes/manager.hpp>
   #include <redGrapes/resource/ioresource.hpp>
   #include <redGrapes/property/inherit.hpp>
   #include <redGrapes/property/resource.hpp>
   #include <redGrapes/property/label.hpp>

   namespace rg = redGrapes;

   using TaskProperties =
       rg::TaskProperties<
           rg::ResourceProperty,
           rg::LabelProperty
       >;

   int main()
   {
       rg::Manager< TaskProperties, rg::ResourceEnqueuePolicy > mgr;

       rg::IOResource a;

       mgr.emplace_task(
           []{ /* ... */ },
           TaskProperties::Builder()
               .label("Task 1")
               .resources({ a.write() })
       );

       mgr.emplace_task(
           []{ /* ... */ },
           TaskProperties::Builder()
               .label("Task 2")
               .resources({ a.read() })
       );

       mgr.emplace_task(
           []{ /* ... */ },
           TaskProperties::Builder()
               .label("Task 3")
               .resources({ a.read() })
       );

       return 0;
   }
