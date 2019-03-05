
#################################
    Describing Resource Usage
#################################

Resource Objects
================

The first thing to do is to represent the resources in your code. A very simple, predefined resource-type is `rmngr::IOResource`, which supports the access modes *read*, *write*, *atomic add* and *atomic mul*.

.. code-block:: c++

    #include <rmngr/resource/ioresource.hpp>

    rmngr::IOResource a;

Resources are copyable:

.. code-block:: c++

    rmngr::IOResource b( a ); // b identifies the same resource as a

If you implement a structure which should be used **as** resource, then just derive from the corresponding resource type:

.. code-block:: c++

    struct MyContainer : rmngr::IOResource {
        // ...
    }

Resource Access
===============

`rmngr::IOResource` also defines convenience wrappers for its access modes, so for example you can call `read()` to create a ResourceAccess which represents a read-acces on the resource.

.. code-block:: c++

    auto access1 = a.read();
    auto access2 = a.write();


Lets suppose your own resource does more than only read/write.
Then you want to define your own AccessPolicy which encodes the possible accesses to your resource type. This implementation must satisfy the :ref:`AccessPolicy concept <concept_AccessPolicy>`.

Consider an array where you can specify, which element you want to access. Two accesses have to be executed sequential, if they use the same index.

.. code-block:: c++

    struct MyArrayAccess {
        int index;

        static bool is_serial(MyArrayAccess a, MyArrayAccess b) {
            return (a.index == b.index);
        }
        static bool is_superset_of(MyArrayAccess a, MyArrayAccesss b) {
            return (a.index == b.index);
        }
    }
    
    struct MyArray : rmngr::Resource<MyArrayAccess> {
        std::array<...> data;

        rmngr::ResourceAccess access_index( int index ) const {
            return this->make_access( MyArrayAccess{ index } );
	}
    }

Resource Users
==============

The goal is to annotate a functor with all its accesses it makes during execution.
For that, every functor is a :ref:`ResourceUser <class_ResourceUser>`, which means it stores a list of :ref:`resource accesses <class_ResourceAccess>`.

.. code-block:: c++

    #include <rmngr/resource/ioresource.hpp>
    #include <rmngr/resource/resource_user.hpp>

    rmngr::IOResource a, b;
    rmngr::ResourceUser user1({ a.read(), b.read() });
    rmngr::ResourceUser user2({ a.read(), b.write() });

    // read-only can be parallel
    rmngr::ResourceUser::is_serial(user1, user1) == true;

    // writes are sequential
    rmngr::ResourceUser::is_serial(user1, user2) == false;
    rmngr::ResourceUser::is_serial(user2, user2) == false;
