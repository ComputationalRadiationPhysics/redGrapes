
cmake_minimum_required(VERSION 3.10.0)

project(redGrapesExamples LANGUAGES CXX)

find_package(redGrapes REQUIRED CONFIG PATHS "${CMAKE_CURRENT_LIST_DIR}/../")
include_directories(SYSTEM ${redGrapes_INCLUDE_DIRS})

find_package(Threads REQUIRED)
find_package(MPI)
find_package(PNGwriter)
find_package(LAPACK)

set(EXAMPLE_NAMES
    1_resources
    2_functors
    3_functors_with_resources
    4_refinements
    5_access_demotion
    6_resource_scope
    7_event
    game_of_life
)

foreach(examplename ${EXAMPLE_NAMES})
    add_executable(${examplename} ${examplename}.cpp)
    target_compile_features(${examplename} PUBLIC cxx_std_14)
    target_link_libraries(${examplename} PRIVATE redGrapes)
    target_link_libraries(${examplename} PRIVATE Threads::Threads)
endforeach()

if(MPI_FOUND)
    add_executable(mpi mpi.cpp)
    target_compile_features(mpi PUBLIC cxx_std_14)
    target_link_libraries(mpi PRIVATE redGrapes)
    target_link_libraries(mpi PRIVATE Threads::Threads)
    target_link_libraries(mpi PRIVATE MPI::MPI_CXX)
endif()

if(LAPACK_FOUND)
    add_executable(cholesky cholesky.cpp)
    target_compile_features(cholesky PUBLIC cxx_std_14)
    target_link_libraries(cholesky PRIVATE redGrapes)
    target_link_libraries(cholesky PRIVATE Threads::Threads)
    target_compile_options(cholesky PRIVATE ${LAPACK_LINKER_FLAGS})
    target_link_libraries(cholesky PRIVATE ${LAPACK_LIBRARIES})
endif()

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    add_executable(cuda_mandelbrot cuda_mandelbrot.cu)
    target_compile_features(cuda_mandelbrot PUBLIC cuda_std_14)
    set_target_properties(cuda_mandelbrot PROPERTIES
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD_REQUIRED ON
        CUDA_LINK_LIBRARIES_KEYWORD PUBLIC
    )
    target_link_libraries(cuda_mandelbrot PRIVATE redGrapes)
    target_link_libraries(cuda_mandelbrot PRIVATE Threads::Threads)
    target_link_libraries(cuda_mandelbrot PRIVATE PNGwriter::PNGwriter)
endif()

