
.. _tutorial_task-properties:

#######################
    Task Properties
#######################

Every task has *properties*, which contain additional scheduling or debug information about that task. What these task-properties are, must be configured by the user.
This is typically done by combining multiple predefined and custom property classes, each providing a *builder*.
RedGrapes provides the means for combining such independent property definitions accordingly from a variadic template:

.. code-block:: c++

   redGrapes::TaskProperties< Property1, Property2, ... >

When creating a task using ``emplace_task()``, the second parameter is the task properties.
Each individual property class should have sensible defaults and provide builder functions for creating property-configurations nicely.

Here is a full example using the predefined ``LabelProperty``:

.. code-block:: c++

   #include <iostream>
   #include <redGrapes/manager.hpp>
   #include <redGrapes/property/label.hpp>

   namespace rg = redGrapes;

   using TaskProperties = rg::TaskProperties< rg::LabelProperty >;

   int main()
   {
       rg::Manager< TaskProperties > mgr;

       mgr.emplace_task(
           [] { std::cout << "Hello World!" << std::endl; },
	   TaskProperties::Builder().label( "Example Task" )
       );

       return 0;
   }


Another essential predefined property is the ``ResourceProperty``, which will be discussed in the next section!
