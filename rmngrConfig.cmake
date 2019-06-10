
cmake_minimum_required(VERSION 3.10.0)

include(CMakeFindDependencyMacro)

set(rmngr_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}")
set(rmngr_INCLUDE_DIRS ${rmngr_INCLUDE_DIRS} "${CMAKE_CURRENT_LIST_DIR}/share/thirdParty/akrzemi/optional/include")
