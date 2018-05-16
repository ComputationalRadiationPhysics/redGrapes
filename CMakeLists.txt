
cmake_minimum_required(VERSION 3.10.0)

project(rmngr)

find_package(Boost 1.62.0 REQUIRED COMPONENTS graph)
find_package(Threads REQUIRED)

add_library(rmngr INTERFACE)
target_compile_features(rmngr INTERFACE
    cxx_std_11
)
target_include_directories(rmngr INTERFACE
    $<BUILD_INTERFACE:${rmngr_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${rmngr_INSTALL_PREFIX}>
)

install(TARGETS rmngr)
install(DIRECTORY rmngr DESTINATION include)

set(EXAMPLE_NAMES
    example
    game_of_life
)

foreach(examplename ${EXAMPLE_NAMES})
    add_executable(${examplename} examples/${examplename}.cpp)
    target_link_libraries(${examplename} PRIVATE rmngr)
    target_link_libraries(${examplename} PRIVATE Threads::Threads)
endforeach()

include(CTest)

# Catch2 for unit tests
if(BUILD_TESTING)
    find_package(Catch2 2.2.1 CONFIG REQUIRED)
    message(STATUS "Catch2: Found version ${Catch2_VERSION}")

    set(TEST_SOURCES
        test/main.cpp
        test/dependency_manager.cpp
        test/access.cpp
        test/resource.cpp
        test/resource_user.cpp)

    set(TEST_TARGET rmngr_test)

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE rmngr)
    add_test(NAME unittest COMMAND ${TEST_TARGET})
endif()

