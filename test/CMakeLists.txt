
cmake_minimum_required(VERSION 3.10.0)

project(rmngrTest)

find_package(rmngr REQUIRED CONFIG PATHS "${CMAKE_CURRENT_LIST_DIR}/..")
include_directories(SYSTEM ${rmngr_INCLUDE_DIRS})

find_package(Catch2 2.2.1 CONFIG)

option(rmngr_USE_INTERNAL_CATCH "Use internally stored Catch2" ON)
option(rmngr_DOWNLOAD_CATCH "Download Catch2 if not found" ON)

include(CTest)
enable_testing()

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
    main.cpp
    dependency_manager.cpp
    access.cpp
    resource.cpp
    resource_user.cpp)

set(TEST_TARGET rmngr_test)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} PRIVATE rmngr)
add_test(NAME unittest COMMAND ${TEST_TARGET})

