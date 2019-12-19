
cmake_minimum_required(VERSION 3.10.0)

project(redGrapesExamples)

find_package(redGrapes REQUIRED CONFIG PATHS "${CMAKE_CURRENT_LIST_DIR}/..")
include_directories(SYSTEM ${redGrapes_INCLUDE_DIRS})

find_package(Threads REQUIRED)
find_package(MPI REQUIRED)

set(EXAMPLE_NAMES
    1_resources
    2_functors
    3_functors_with_resources
    4_refinements
    5_access_demotion
    6_resource_scope
    7_event
    game_of_life
    mpi
)

foreach(examplename ${EXAMPLE_NAMES})
    add_executable(${examplename} ${examplename}.cpp)
    target_link_libraries(${examplename} PRIVATE redGrapes)
    target_link_libraries(${examplename} PRIVATE Threads::Threads)
endforeach()

target_link_libraries(mpi PRIVATE MPI::MPI_CXX)


