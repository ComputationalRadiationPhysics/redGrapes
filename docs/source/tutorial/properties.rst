
.. _tutorial_task-properties:

#######################
    Task Properties
#######################

TODO: redGrapes_config.hpp

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
   #include <redGrapes/redGrapes.hpp>
   #include <redGrapes/property/label.hpp>

   namespace rg = redGrapes;

   int main()
   {
       rg::init();

       rg::emplace_task(
           [] { std::cout << "Hello World!" << std::endl; },
       ).label( "Example Task" );

       rg::finalize();

       return 0;
   }


Another essential predefined property is the ``ResourceProperty``, which will be discussed in the next section!
