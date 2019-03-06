
################
    Building
################

Requirements
============

- C++14
- Boost Graph
- Boost MPL

Build a Project using rmngr
===========================
rmngr is a C++ header-only library so you only need to set the include path.
If you are using CMake, the following is sufficient:
::

    find_package(rmngr REQUIRED CONFIG PATHS "[path to rmngr]")
    include_directories(SYSTEM ${rmngr_INCLUDE_DIRS})
