
cmake_minimum_required(VERSION 3.10.0)

project(rmngrExamples)

find_package(rmngr REQUIRED CONFIG PATHS "${CMAKE_CURRENT_LIST_DIR}/..")
include_directories(SYSTEM ${rmngr_INCLUDE_DIRS})

find_package(Threads REQUIRED)

set(EXAMPLE_NAMES
    1_resources
    2_functors
    3_functors_with_resources
    4_refinements
    5_access_demotion
    6_dynamic_properties
    7_scheduler_singleton
    game_of_life
)

foreach(examplename ${EXAMPLE_NAMES})
    add_executable(${examplename} ${examplename}.cpp)
    target_link_libraries(${examplename} PRIVATE rmngr)
    target_link_libraries(${examplename} PRIVATE Threads::Threads)
endforeach()

