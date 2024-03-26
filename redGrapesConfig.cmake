cmake_minimum_required(VERSION 3.18.0)

include(CMakeFindDependencyMacro)

project(redGrapes VERSION 0.1.0)

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

set(redGrapes_CXX_STANDARD_DEFAULT "20")
# Check whether redGrapes_CXX_STANDARD has already been defined as a non-cached variable.
if(DEFINED redGrapes)
  set(redGrapes_CXX_STANDARD_DEFAULT ${redGrapes_CXX_STANDARD})
endif()

set(redGrapes_CXX_STANDARD ${redGrapes_CXX_STANDARD_DEFAULT} CACHE STRING "C++ standard version")
set_property(CACHE redGrapes_CXX_STANDARD PROPERTY STRINGS "20;23")

if( NOT TARGET redGrapes )
  add_library(redGrapes INTERFACE)
  target_compile_features(redGrapes INTERFACE cxx_std_${redGrapes_CXX_STANDARD})
  if(MSVC)
    target_compile_options(redGrapes INTERFACE /W4 /WX)
  else()
    target_compile_options(redGrapes INTERFACE -Wall -Wextra -Wpedantic)
endif()

endif()

target_include_directories(redGrapes INTERFACE
    $<BUILD_INTERFACE:${redGrapes_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${redGrapes_INSTALL_PREFIX}>
)

target_link_libraries(redGrapes INTERFACE ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(redGrapes INTERFACE ${Boost_LIBRARIES})
target_link_libraries(redGrapes INTERFACE fmt::fmt)
target_link_libraries(redGrapes INTERFACE spdlog::spdlog)
target_link_libraries(redGrapes INTERFACE ${HWLOC})
set(redGrapes_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})
set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} "${CMAKE_CURRENT_LIST_DIR}/share/thirdParty/cameron314/concurrentqueue/include")
set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} ${HWLOC_INCLUDE_DIR})

set(redGrapes_LIBRARIES ${Boost_LIBRARIES} fmt::fmt spdlog::spdlog ${CMAKE_THREAD_LIBS_INIT} ${HWLOC})

option(redGrapes_ENABLE_BACKWARDCPP "Enable extended debugging with `backward-cpp`" OFF)
option(redGrapes_ENABLE_PERFETTO "Enable tracing support with perfetto" OFF)

if(redGrapes_ENABLE_BACKWARDCPP)
  set(Backward_DIR "${CMAKE_CURRENT_LIST_DIR}/share/thirdParty/bombela/backward-cpp")
  find_package(Backward)

  add_compile_definitions(REDGRAPES_ENABLE_BACKWARDCPP=1)
  target_link_libraries(redGrapes INTERFACE Backward::Backward)
endif()

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

target_include_directories(redGrapes INTERFACE ${redGrapes_INCLUDE_DIRS})

