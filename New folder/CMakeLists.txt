cmake_minimum_required(VERSION 3.18)
project(ImageEnhancement LANGUAGES CXX CUDA)

# Set CMake policy to suppress FindCUDA warning
cmake_policy(SET CMP0146 NEW)

# Use C++14 and CUDA 14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# Detect CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# Help NVCC locate MSVC compiler (required for CUDA >= 12.x)
if (MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    set(CMAKE_CUDA_HOST_COMPILER 
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64/cl.exe")
endif()

# Silence deprecated GPU target warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# Find packages
find_package(MPI REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)

# Common sources
set(COMMON_SOURCES image_utils.cpp)

# Sequential version
add_executable(sequential ${COMMON_SOURCES} main.cpp)
target_compile_definitions(sequential PRIVATE USE_SEQUENTIAL)

# OpenMP version
add_executable(omp_version ${COMMON_SOURCES} main.cpp omp_image_utils.cpp)
target_compile_definitions(omp_version PRIVATE USE_OMP)
target_link_libraries(omp_version PRIVATE OpenMP::OpenMP_CXX)

# MPI version
add_executable(mpi_version ${COMMON_SOURCES} main.cpp mpi_image_utils.cpp)
target_compile_definitions(mpi_version PRIVATE USE_MPI)
target_include_directories(mpi_version PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(mpi_version PRIVATE ${MPI_CXX_LIBRARIES})

# CUDA version
add_executable(cuda_version ${COMMON_SOURCES} main.cpp cuda_kernels.cu cuda_image_utils.cu)
target_compile_definitions(cuda_version PRIVATE USE_CUDA)
set_target_properties(cuda_version PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "75"
)

find_package(CUDAToolkit REQUIRED)

# Set output folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
