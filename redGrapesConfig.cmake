
cmake_minimum_required(VERSION 3.10.0)

include(CMakeFindDependencyMacro)

find_package(Boost 1.62.0 REQUIRED COMPONENTS graph context)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

set(redGrapes_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}")
set(redGrapes_INCLUDE_DIRS ${redGrapes_INCLUDE_DIRS} "${CMAKE_CURRENT_LIST_DIR}/share/thirdParty/akrzemi/optional/include")
set(redGrapes_LIBRARIES ${Boost_LIBRARIES})
