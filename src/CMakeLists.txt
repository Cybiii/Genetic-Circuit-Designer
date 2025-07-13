# Source CMakeLists.txt

# Core library
add_library(circuit_core
    core/circuit.cpp
    core/circuit_simulator.cpp
    core/circuit_builder.cpp
    core/test_generators.cpp
)

target_include_directories(circuit_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(circuit_core
    circuit_utils
)

# GPU library
add_library(circuit_gpu
    gpu/gpu_simulator.cpp
    gpu/gpu_memory.cpp
    gpu/circuit_kernels.cu
    gpu/genetic_kernels.cu
    gpu/cuda_utils.cpp
)

target_include_directories(circuit_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(circuit_gpu
    circuit_core
    circuit_utils
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_CURAND_LIBRARIES}
)

# Set CUDA properties
set_target_properties(circuit_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Genetic Algorithm library
add_library(circuit_ga
    ga/genome.cpp
    ga/genetic_algorithm.cpp
    ga/selection.cpp
    ga/crossover.cpp
    ga/mutation.cpp
    ga/fitness_evaluation.cpp
    ga/population.cpp
    ga/evolution_strategies.cpp
)

target_include_directories(circuit_ga PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(circuit_ga
    circuit_core
    circuit_utils
)

# Conditionally link GPU support if available
if(CUDA_FOUND)
    target_link_libraries(circuit_ga circuit_gpu)
    target_compile_definitions(circuit_ga PRIVATE CUDA_ENABLED)
endif()

# Visualization library
add_library(circuit_viz
    viz/visualization.cpp
    viz/circuit_renderer.cpp
    viz/evolution_visualizer.cpp
    viz/interactive_builder.cpp
    viz/color_schemes.cpp
    viz/opengl_utils.cpp
    viz/imgui_integration.cpp
)

target_include_directories(circuit_viz PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(circuit_viz
    circuit_core
    circuit_ga
    circuit_utils
    ${OPENGL_LIBRARIES}
    glfw
    ${CMAKE_DL_LIBS}
)

# Utils library
add_library(circuit_utils
    utils/logger.cpp
    utils/profiler.cpp
    utils/file_utils.cpp
    utils/string_utils.cpp
    utils/math_utils.cpp
    utils/memory_utils.cpp
    utils/config_manager.cpp
    utils/error_handler.cpp
    utils/thread_utils.cpp
    utils/circuit_utils.cpp
)

target_include_directories(circuit_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(circuit_utils ws2_32)
elseif(UNIX)
    target_link_libraries(circuit_utils pthread)
endif()

# Set library properties
set_target_properties(circuit_core circuit_ga circuit_viz circuit_utils PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Install targets
install(TARGETS circuit_core circuit_ga circuit_viz circuit_utils
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Create main executable target in parent scope
set(CIRCUIT_LIBRARIES 
    circuit_core
    circuit_gpu
    circuit_ga
    circuit_viz
    circuit_utils
    PARENT_SCOPE
) 