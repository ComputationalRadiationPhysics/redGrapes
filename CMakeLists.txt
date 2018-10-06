
cmake_minimum_required(VERSION 3.10.0)

project(rmngr)

find_package(Boost 1.62.0 REQUIRED COMPONENTS graph)

add_library(rmngr INTERFACE)
target_compile_features(rmngr INTERFACE
    cxx_std_11
)
target_include_directories(rmngr INTERFACE
    $<BUILD_INTERFACE:${rmngr_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${rmngr_INSTALL_PREFIX}>
)

option(rmngr_BUILD_EXAMPLES "Build the examples" ON)

if(rmngr_BUILD_EXAMPLES)
    add_subdirectory("examples/")
endif()

if(BUILD_TESTING)
    add_subdirectory("test/")
endif()

