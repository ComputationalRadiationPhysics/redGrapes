
Requirements
============

- C++14
- Boost >= 1.62

Build a Project using RedGrapes
===============================
RedGrapes is a C++ header-only library so you only need to set the include path.
If you are using CMake, the following is sufficient:
::

    find_package(redGrapes REQUIRED CONFIG PATHS "[path to redGrapes]")
    include_directories(SYSTEM ${redGrapes_INCLUDE_DIRS})

Examples & Tests
================

In order to build the examples and tests, do the typical cmake procedure:
::
    mkdir build
    cd build
    cmake ..
    make -j
