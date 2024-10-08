cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(uncc_sar_focusing LANGUAGES CXX)

set( CMAKE_VERBOSE_MAKEFILE ON)

#set( Matlab_ROOT_DIR "/usr/local/bin/matlab/R2020a")
set( Matlab_ROOT_DIR "/usr/local/bin/matlab/R2021a")

# Code should build on CUDA version 11.0 - 11.5 not currently compatible with CUDA 11.6
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.5")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.5/bin/nvcc")
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

set(CMAKE_DEFAULT_CUDA_ARCHITECTURES 50 52 60 61 75 86)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

foreach (OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PROJECT_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PROJECT_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${PROJECT_BINARY_DIR}/lib)
endforeach()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake" CACHE STRING "Modules for CMake" FORCE)

set(MATLAB_PROJECT_DIR "${CMAKE_CURRENT_LIST_DIR}")

add_subdirectory(third_party)
include_directories(
    ./include
    ./include/third_party
    third_party/matio/src
    third_party/cuGridSearch/include
    ${PROJECT_BINARY_DIR}/third_party/matio/src
)
link_directories(
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
)

add_subdirectory(src)

