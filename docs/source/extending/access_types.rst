
###################################
    Creating new Resource-Types
###################################

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


Combining Access Types
======================
TODO
