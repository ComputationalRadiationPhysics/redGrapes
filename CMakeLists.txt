
cmake_minimum_required(VERSION 3.10.0)

project(rmngr)

option(rmngr_USE_INTERNAL_CATCH "Use internally stored Catch2" ON)
option(rmngr_DOWNLOAD_CATCH "Download Catch2 if not found" ON)

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
    1_resources
    2_functors
    3_functors_with_resources
    4_refinements
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
    enable_testing()

    find_package(Catch2 2.2.1 CONFIG)
    if(Catch2_FOUND)
        message(STATUS "Catch2: Found version ${Catch2_VERSION}")
    elseif(rmngr_USE_INTERNAL_CATCH)
        set(Catch2_DIR ${rmngr_SOURCE_DIR}/share/thirdParty/catch2)
        if(EXISTS ${Catch2_DIR})
            message(STATUS "Catch2: found INTERNAL version")
        else()
            if(rmngr_DOWNLOAD_CATCH)
                message(STATUS "Catch2: Downloading latest version...")
                execute_process(COMMAND mkdir -p "${Catch2_DIR}/include/catch")
                execute_process(
                  WORKING_DIRECTORY "${Catch2_DIR}/include/catch"
                  COMMAND curl -O "https://raw.githubusercontent.com/catchorg/Catch2/master/single_include/catch2/catch.hpp")
            else()
                message(WARNING "Catch2: not found and download is disabled")
            endif()
        endif()

        target_include_directories(rmngr SYSTEM INTERFACE
            $<BUILD_INTERFACE:${Catch2_DIR}/include>
            )
    else()
        message(WARNING "Catch2: No CMake package or internal version found")
    endif()

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

