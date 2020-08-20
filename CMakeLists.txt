
cmake_minimum_required(VERSION 3.10.0)

project(redGrapes VERSION 0.1.0)

find_package(Boost 1.62.0 REQUIRED COMPONENTS graph context)

add_library(redGrapes INTERFACE)
target_compile_features(redGrapes INTERFACE
    cxx_std_14
)
target_include_directories(redGrapes INTERFACE
    $<BUILD_INTERFACE:${redGrapes_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${redGrapes_INSTALL_PREFIX}>
)

target_link_libraries(redGrapes INTERFACE ${Boost_LIBRARIES})

########################################################
#  Examples & Tests
########################################################
option(redGrapes_BUILD_EXAMPLES "Build the examples" ON)

if(redGrapes_BUILD_EXAMPLES)
    add_subdirectory("examples/")
endif()

if(BUILD_TESTING)
    add_subdirectory("test/")
endif()

#########################################################
#  Installation
#########################################################
option(redGrapes_USE_INTERNAL_OPTIONAL "use akrzemi's optional library for c++14" ON)

include(GNUInstallDirs)
set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/redGrapes"
    CACHE PATH "CMake config package location for installed targets")
if(WIN32)
    set_property(CACHE CMAKE_INSTALL_CMAKEDIR PROPERTY VALUE "cmake")
endif()

install(DIRECTORY "${redGrapes_SOURCE_DIR}/redGrapes"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# install third-party libraries
if(redGrapes_USE_INTERNAL_OPTIONAL)
    install(DIRECTORY "${redGrapes_SOURCE_DIR}/share/thirdParty/akrzemi/optional/include/akrzemi"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

# Version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file("redGrapesConfigVersion.cmake"
    VERSION ${redGrapes_VERSION}
    COMPATIBILITY SameMajorVersion
)

# CMake package file for find_package() in depending projects
install(
    FILES
        ${redGrapes_SOURCE_DIR}/redGrapesConfig.cmake
        ${redGrapes_BINARY_DIR}/redGrapesConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_CMAKEDIR}
)
