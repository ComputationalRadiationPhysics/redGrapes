
####################
   Initialization
####################

Before you can submit tasks to the redGrapes runtime, you need to call,  ``redGrapes::init()``.


.. code-block:: c++

   #include <redGrapes/redGrapes.hpp>

   int main()
   {
       redGrapes::init();

       // ... create tasks ...
       
       redGrapes::finalize();
       return 0;
   }



TODO: redGrapes::barrier()
TODO: redGrapes_config.h
