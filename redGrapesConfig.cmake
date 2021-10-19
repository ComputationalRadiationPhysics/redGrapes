cmake_minimum_required(VERSION 3.10.0)

include(CMakeFindDependencyMacro)

project(redGrapes VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 14)

find_package(Boost 1.62.0 REQUIRED COMPONENTS context)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

add_library(redGrapes INTERFACE)
target_compile_features(redGrapes INTERFACE
    cxx_std_14
)
target_include_directories(redGrapes INTERFACE
    $<BUILD_INTERFACE:${redGrapes_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${redGrapes_INSTALL_PREFIX}>
)

target_link_libraries(redGrapes INTERFACE ${Boost_LIBRARIES})
target_link_libraries(redGrapes INTERFACE fmt::fmt)
target_link_libraries(redGrapes INTERFACE spdlog::spdlog)

set(redGrapes_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}")
set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} "${CMAKE_CURRENT_LIST_DIR}/share/thirdParty/akrzemi/optional/include")
set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} "${CMAKE_CURRENT_LIST_DIR}/share/thirdParty/cameron314/concurrentqueue/include")
set(redGrapes_LIBRARIES ${Boost_LIBRARIES} fmt::fmt spdlog::spdlog)

