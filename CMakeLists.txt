cmake_minimum_required(VERSION 3.18)
project(GeneticCircuitDesigner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optional visualization support
option(ENABLE_VISUALIZATION "Enable OpenGL visualization" OFF)
option(ENABLE_GPU "Enable GPU acceleration" ON)

# Find required packages
if(ENABLE_GPU)
    find_package(CUDA REQUIRED)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# Find optional packages
if(ENABLE_VISUALIZATION)
    find_package(OpenGL)
    find_package(glfw3)
    find_package(PkgConfig)
endif()

# Compiler flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
endif()

if(ENABLE_GPU)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/nlohmann
)

if(ENABLE_VISUALIZATION)
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    )
endif()

if(ENABLE_GPU)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
# add_subdirectory(examples)  # Temporarily disabled for core testing
add_subdirectory(external)

# Main executable
add_executable(circuit_designer
    src/main.cpp
)

# Link libraries
set(CORE_LIBS
    circuit_core
    circuit_ga
    circuit_utils
)

if(ENABLE_GPU)
    list(APPEND CORE_LIBS circuit_gpu)
endif()

if(ENABLE_VISUALIZATION)
    list(APPEND CORE_LIBS circuit_viz)
endif()

target_link_libraries(circuit_designer ${CORE_LIBS})

if(ENABLE_GPU)
    target_link_libraries(circuit_designer ${CUDA_LIBRARIES})
endif()

if(ENABLE_VISUALIZATION AND OpenGL_FOUND AND glfw3_FOUND)
    target_link_libraries(circuit_designer ${OPENGL_LIBRARIES} glfw)
endif()

# Define preprocessor macros
if(ENABLE_VISUALIZATION)
    target_compile_definitions(circuit_designer PRIVATE ENABLE_VISUALIZATION)
endif()

if(ENABLE_GPU)
    target_compile_definitions(circuit_designer PRIVATE ENABLE_GPU)
endif()

# Installation
install(TARGETS circuit_designer DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY data/ DESTINATION share/circuit_designer) 

# CPU Benchmark executable
add_executable(cpu_benchmark ../cpu_benchmark.cpp)
target_include_directories(cpu_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(cpu_benchmark) 

# Add CUDA device query executable if CUDA is available
if(CUDA_FOUND)
    add_executable(gpu_device_query ../gpu_device_query.cpp)
    target_link_libraries(gpu_device_query ${CUDA_LIBRARIES})
    set_target_properties(gpu_device_query PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif() 