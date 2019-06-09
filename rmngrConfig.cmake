
cmake_minimum_required(VERSION 3.10.0)

include(CMakeFindDependencyMacro)

target_include_directories(rmngr SYSTEM INTERFACE
    $<BUILD_INTERFACE:${rmngr_SOURCE_DIR}/share/thirdParty/akrzemi/optional/include>
)

set(rmngr_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}")

