
cmake_minimum_required(VERSION 3.10.0)

project(redGrapes VERSION 0.1.0)

find_package(Boost 1.62.0 REQUIRED COMPONENTS graph)

add_library(redGrapes INTERFACE)
target_compile_features(redGrapes INTERFACE
    cxx_std_14
)
target_include_directories(redGrapes INTERFACE
    $<BUILD_INTERFACE:${redGrapes_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${redGrapes_INSTALL_PREFIX}>
)

option(redGrapes_BUILD_EXAMPLES "Build the examples" ON)

if(redGrapes_BUILD_EXAMPLES)
    add_subdirectory("examples/")
endif()

if(BUILD_TESTING)
    add_subdirectory("test/")
endif()

