cmake_minimum_required(VERSION 3.18.0)

include(CMakeFindDependencyMacro)

project(redGrapes VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 14)

find_package(Boost 1.62.0 REQUIRED COMPONENTS context)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

if( NOT TARGET redGrapes )
add_library(redGrapes
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/resource/resource.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/dispatch/thread/execute.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/scheduler/event.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/scheduler/event_ptr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/task/property/graph.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/task/task_space.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/task/queue.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/util/chunk_allocator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/util/chunked_list.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/util/trace.cpp
  ${CMAKE_CURRENT_LIST_DIR}/redGrapes/redGrapes.cpp
)
endif()

target_compile_features(redGrapes PUBLIC
    cxx_std_14
)

add_compile_definitions(PERFETTO_ALLOW_SUB_CPP17)

target_include_directories(redGrapes PUBLIC
    $<BUILD_INTERFACE:${redGrapes_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${redGrapes_INSTALL_PREFIX}>
)

#target_include_directories(redGrapes PUBLIC perfetto/sdk)
add_library(perfetto STATIC /usr/share/perfetto/sdk/perfetto.cc)

target_link_libraries(redGrapes PUBLIC ${Boost_LIBRARIES})
target_link_libraries(redGrapes PUBLIC fmt::fmt)
target_link_libraries(redGrapes PUBLIC spdlog::spdlog)
target_link_libraries(redGrapes PUBLIC perfetto ${CMAKE_THREAD_LIBS_INIT})

set(redGrapes_INCLUDE_DIRS ${redGrapes_CONFIG_INCLUDE_DIR} ${CMAKE_CURRENT_LIST_DIR})
set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} "${CMAKE_CURRENT_LIST_DIR}/share/thirdParty/akrzemi/optional/include")
set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} "${CMAKE_CURRENT_LIST_DIR}/share/thirdParty/cameron314/concurrentqueue/include")
set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} "/usr/share/perfetto/sdk")
set(redGrapes_LIBRARIES ${Boost_LIBRARIES} fmt::fmt spdlog::spdlog perfetto ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(redGrapes PUBLIC ${redGrapes_INCLUDE_DIRS})

