cmake_minimum_required(VERSION 3.18.0)

include(CMakeFindDependencyMacro)

project(redGrapes VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 14)

find_package(Boost 1.62.0 REQUIRED COMPONENTS context)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)


## Find HwLoc
find_path(HWLOC_INCLUDE_DIR
  NAMES
  hwloc.h
  PATHS
  /opt/local
  /usr/local
  /usr
  ENV "PROGRAMFILES(X86)"
  ENV "HWLOC_ROOT"
  PATH_SUFFIXES
  include)

find_library(HWLOC
  NAMES
  libhwloc.lib
  hwloc
  PATHS
  ENV "HWLOC_ROOT"
  PATH_SUFFIXES
  lib)

if(HWLOC STREQUAL "HWLOC-NOTFOUND" OR ${HWLOC_INCLUDE_DIR} STREQUAL "HWLOC_INCLUDE_DIR-NOTFOUND")
  message(FATAL_ERROR "hwloc NOT found: use `-DHWLOC_ENABLE=OFF` to build without hwloc support")

else()
  message(STATUS "Found hwloc")
endif()


if( NOT TARGET redGrapes )
add_library(redGrapes
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/resource/resource.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/dispatch/thread/execute.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/dispatch/thread/cpuset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/dispatch/thread/worker.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/dispatch/thread/worker_pool.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/scheduler/event.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/scheduler/event_ptr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/task/property/graph.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/task/task_space.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/task/queue.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/util/allocator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/util/bump_alloc_chunk.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/util/trace.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/redGrapes.cpp
)
endif()

target_compile_features(redGrapes PUBLIC
    cxx_std_14
)

target_include_directories(redGrapes PUBLIC
    $<BUILD_INTERFACE:${redGrapes_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${redGrapes_INSTALL_PREFIX}>
)

target_link_libraries(redGrapes PUBLIC ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(redGrapes PUBLIC ${Boost_LIBRARIES})
target_link_libraries(redGrapes PUBLIC fmt::fmt)
target_link_libraries(redGrapes PUBLIC spdlog::spdlog)
target_link_libraries(redGrapes PUBLIC ${HWLOC})

set(redGrapes_INCLUDE_DIRS ${redGrapes_CONFIG_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR})
set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} "${CMAKE_CURRENT_LIST_DIR}/share/thirdParty/akrzemi/optional/include")
set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} "${CMAKE_CURRENT_LIST_DIR}/share/thirdParty/cameron314/concurrentqueue/include")
set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} ${HWLOC_INCLUDE_DIR})

set(redGrapes_LIBRARIES ${Boost_LIBRARIES} fmt::fmt spdlog::spdlog ${CMAKE_THREAD_LIBS_INIT} ${HWLOC})

option(redGrapes_ENABLE_PERFETTO "Enable tracing support with perfetto" OFF)

if(redGrapes_ENABLE_PERFETTO)
    add_compile_definitions(PERFETTO_ALLOW_SUB_CPP17)
    add_compile_definitions(REDGRAPES_ENABLE_TRACE=1)

    if( NOT TARGET perfetto )
      add_library(perfetto STATIC /usr/share/perfetto/sdk/perfetto.cc)
    endif()

    target_link_libraries(redGrapes PUBLIC perfetto)
    set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} "/usr/share/perfetto/sdk")
    set(redGrapes_LIBRARIES ${Boost_LIBRARIES} fmt::fmt spdlog::spdlog perfetto ${CMAKE_THREAD_LIBS_INIT})
endif()

target_include_directories(redGrapes PUBLIC ${redGrapes_INCLUDE_DIRS})

