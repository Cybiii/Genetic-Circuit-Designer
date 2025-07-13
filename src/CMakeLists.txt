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

# GPU library (optional)
if(ENABLE_GPU)
    add_library(circuit_gpu
        gpu/gpu_simulator.cpp
        gpu/gpu_memory.cpp
        gpu/circuit_kernels.cu
        gpu/genetic_kernels.cu
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
endif()

# Genetic Algorithm library
add_library(circuit_ga
    ga/genome.cpp
    ga/genetic_algorithm.cpp
    ga/selection.cpp
    ga/crossover.cpp
    ga/mutation.cpp
    ga/fitness_evaluation.cpp
)

target_include_directories(circuit_ga PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(circuit_ga
    circuit_core
    circuit_utils
)

# Conditionally link GPU support if available
if(ENABLE_GPU)
    target_link_libraries(circuit_ga circuit_gpu)
    target_compile_definitions(circuit_ga PRIVATE CUDA_ENABLED)
endif()

# Visualization library (optional)
if(ENABLE_VISUALIZATION)
    add_library(circuit_viz
        viz/visualization.cpp
        viz/circuit_renderer.cpp
        viz/evolution_visualizer.cpp
        viz/opengl_utils.cpp
    )

    target_include_directories(circuit_viz PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )

    target_link_libraries(circuit_viz
        circuit_core
        circuit_ga
        circuit_utils
    )

    # Link OpenGL libraries if available
    if(OpenGL_FOUND)
        target_link_libraries(circuit_viz ${OPENGL_LIBRARIES})
    endif()
    
    if(glfw3_FOUND)
        target_link_libraries(circuit_viz glfw)
    endif()
endif()

# Utils library
add_library(circuit_utils
    utils/logger.cpp
    utils/profiler.cpp
    utils/file_utils.cpp
    utils/math_utils.cpp
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

# Set library properties for always-built targets
set_target_properties(circuit_core circuit_ga circuit_utils PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Set properties for optional targets
if(ENABLE_GPU)
    set_target_properties(circuit_gpu PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

if(ENABLE_VISUALIZATION)
    set_target_properties(circuit_viz PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Install targets
set(INSTALL_TARGETS circuit_core circuit_ga circuit_utils)

if(ENABLE_GPU)
    list(APPEND INSTALL_TARGETS circuit_gpu)
endif()

if(ENABLE_VISUALIZATION)
    list(APPEND INSTALL_TARGETS circuit_viz)
endif()

install(TARGETS ${INSTALL_TARGETS}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Create main executable target in parent scope
set(CIRCUIT_LIBRARIES 
    circuit_core
    circuit_ga
    circuit_utils
    PARENT_SCOPE
)

if(ENABLE_GPU)
    set(CIRCUIT_LIBRARIES ${CIRCUIT_LIBRARIES} circuit_gpu PARENT_SCOPE)
endif()

if(ENABLE_VISUALIZATION)
    set(CIRCUIT_LIBRARIES ${CIRCUIT_LIBRARIES} circuit_viz PARENT_SCOPE)
endif() 