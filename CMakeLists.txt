cmake_minimum_required(VERSION 3.18)
project(GeneticCircuitDesigner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(PkgConfig REQUIRED)

# Set CUDA properties
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/external/nlohmann
    ${CUDA_INCLUDE_DIRS}
)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(external)

# Main executable
add_executable(circuit_designer
    src/main.cpp
)

target_link_libraries(circuit_designer
    circuit_core
    circuit_gpu
    circuit_ga
    circuit_viz
    circuit_utils
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    glfw
    ${CMAKE_DL_LIBS}
)

# Installation
install(TARGETS circuit_designer DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY data/ DESTINATION share/circuit_designer) 