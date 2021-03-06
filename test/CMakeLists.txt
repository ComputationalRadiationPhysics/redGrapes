
cmake_minimum_required(VERSION 3.10.0)

project(redGrapesTest)

find_package(redGrapes REQUIRED CONFIG PATHS "${CMAKE_CURRENT_LIST_DIR}/..")
include_directories(SYSTEM ${redGrapes_INCLUDE_DIRS})

find_package(Threads REQUIRED)

find_package(Catch2 2.2.1 CONFIG)

option(redGrapes_USE_INTERNAL_CATCH "Use internally stored Catch2" ON)
option(redGrapes_DOWNLOAD_CATCH "Download Catch2 if not found" ON)

include(CTest)
enable_testing()

if(Catch2_FOUND)
    message(STATUS "Catch2: Found version ${Catch2_VERSION}")
elseif(redGrapes_USE_INTERNAL_CATCH)
    set(Catch2_DIR ${redGrapes_SOURCE_DIR}/share/thirdParty/catch2)
    if(EXISTS ${Catch2_DIR})
        message(STATUS "Catch2: found INTERNAL version")
    else()
        if(redGrapes_DOWNLOAD_CATCH)
            message(STATUS "Catch2: Downloading latest version...")
            execute_process(COMMAND mkdir -p "${Catch2_DIR}/include/catch")
            execute_process(
              WORKING_DIRECTORY "${Catch2_DIR}/include/catch"
              COMMAND curl -LO "https://github.com/catchorg/Catch2/releases/download/v2.13.6/catch.hpp")
        else()
            message(WARNING "Catch2: not found and download is disabled")
        endif()
    endif()

    target_include_directories(redGrapes SYSTEM INTERFACE
        $<BUILD_INTERFACE:${Catch2_DIR}/include>
    )
else()
    message(WARNING "Catch2: No CMake package or internal version found")
endif()



### The following is borrowed from:
### https://github.com/google/boringssl/blob/master/CMakeLists.txt

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CLANG 1)
endif()

if(MSAN)
  if(ASAN)
    message(FATAL_ERROR "ASAN and MSAN are mutually exclusive")
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=memory -fsanitize-memory-track-origins -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fsanitize-memory-track-origins -fno-omit-frame-pointer")
  set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -fsanitize=memory -fsanitize-memory-track-origins -fno-omit-frame-pointer")
endif()

if(ASAN)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer")
endif()

if(CFI)
  if(NOT CLANG)
    message(FATAL_ERROR "Cannot enable CFI unless using Clang")
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=cfi -fno-sanitize-trap=cfi -flto=thin")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=cfi -fno-sanitize-trap=cfi -flto=thin")
endif()

if(TSAN)
  if(NOT CLANG)
    message(FATAL_ERROR "Cannot enable TSAN unless using Clang")
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
endif()

if(UBSAN)
  if(NOT CLANG)
    message(FATAL_ERROR "Cannot enable UBSAN unless using Clang")
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=undefined")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")

  if(NOT UBSAN_RECOVER)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-sanitize-recover=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize-recover=undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-sanitize-recover=undefined")
  endif()
endif()


set(TEST_SOURCES
    main.cpp
    dependency_manager.cpp
    access.cpp
    resource.cpp
    resource_user.cpp)

set(TEST_TARGET redGrapes_test)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} PRIVATE redGrapes)
target_link_libraries(${TEST_TARGET} PRIVATE Threads::Threads)
add_test(NAME unittest COMMAND ${TEST_TARGET})

