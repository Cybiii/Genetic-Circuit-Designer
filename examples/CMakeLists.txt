# Examples CMakeLists.txt

# Basic examples
add_executable(basic_circuit_example
    basic/circuit_creation.cpp
)

target_include_directories(basic_circuit_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(basic_circuit_example
    circuit_core
    circuit_utils
)

add_executable(basic_genetic_example
    basic/genetic_algorithm.cpp
)

target_include_directories(basic_genetic_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(basic_genetic_example
    circuit_core
    circuit_ga
    circuit_utils
)

add_executable(basic_visualization_example
    basic/visualization.cpp
)

target_include_directories(basic_visualization_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(basic_visualization_example
    circuit_core
    circuit_viz
    circuit_utils
)

# Advanced examples
add_executable(advanced_gpu_example
    advanced/gpu_acceleration.cpp
)

target_include_directories(advanced_gpu_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(advanced_gpu_example
    circuit_core
    circuit_ga
    circuit_utils
)

# Conditionally link GPU library if available
if(CUDA_FOUND)
    target_link_libraries(advanced_gpu_example circuit_gpu)
    target_compile_definitions(advanced_gpu_example PRIVATE CUDA_ENABLED)
endif()

add_executable(advanced_evolution_example
    advanced/evolution_strategies.cpp
)

target_include_directories(advanced_evolution_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(advanced_evolution_example
    circuit_core
    circuit_ga
    circuit_utils
)

add_executable(advanced_benchmark_example
    advanced/performance_benchmark.cpp
)

target_include_directories(advanced_benchmark_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(advanced_benchmark_example
    circuit_core
    circuit_ga
    circuit_utils
)

# Conditionally link GPU library if available
if(CUDA_FOUND)
    target_link_libraries(advanced_benchmark_example circuit_gpu)
endif()

add_executable(advanced_interactive_example
    advanced/interactive_design.cpp
)

target_include_directories(advanced_interactive_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(advanced_interactive_example
    circuit_core
    circuit_viz
    circuit_utils
)

# Tutorial examples
add_executable(tutorial_01_hello_circuit
    tutorials/01_hello_circuit.cpp
)

target_include_directories(tutorial_01_hello_circuit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(tutorial_01_hello_circuit
    circuit_core
    circuit_utils
)

add_executable(tutorial_02_circuit_simulation
    tutorials/02_circuit_simulation.cpp
)

target_include_directories(tutorial_02_circuit_simulation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(tutorial_02_circuit_simulation
    circuit_core
    circuit_utils
)

add_executable(tutorial_03_genetic_algorithm
    tutorials/03_genetic_algorithm.cpp
)

target_include_directories(tutorial_03_genetic_algorithm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(tutorial_03_genetic_algorithm
    circuit_core
    circuit_ga
    circuit_utils
)

add_executable(tutorial_04_gpu_acceleration
    tutorials/04_gpu_acceleration.cpp
)

target_include_directories(tutorial_04_gpu_acceleration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(tutorial_04_gpu_acceleration
    circuit_core
    circuit_ga
    circuit_utils
)

if(CUDA_FOUND)
    target_link_libraries(tutorial_04_gpu_acceleration circuit_gpu)
    target_compile_definitions(tutorial_04_gpu_acceleration PRIVATE CUDA_ENABLED)
endif()

add_executable(tutorial_05_visualization
    tutorials/05_visualization.cpp
)

target_include_directories(tutorial_05_visualization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(tutorial_05_visualization
    circuit_core
    circuit_viz
    circuit_utils
)

# Set properties for all examples
set_target_properties(
    basic_circuit_example
    basic_genetic_example
    basic_visualization_example
    advanced_gpu_example
    advanced_evolution_example
    advanced_benchmark_example
    advanced_interactive_example
    tutorial_01_hello_circuit
    tutorial_02_circuit_simulation
    tutorial_03_genetic_algorithm
    tutorial_04_gpu_acceleration
    tutorial_05_visualization
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Create custom targets for running examples
add_custom_target(run_basic_examples
    COMMAND echo "Running basic examples..."
    COMMAND basic_circuit_example
    COMMAND basic_genetic_example
    COMMAND basic_visualization_example
    DEPENDS basic_circuit_example basic_genetic_example basic_visualization_example
)

add_custom_target(run_advanced_examples
    COMMAND echo "Running advanced examples..."
    COMMAND advanced_gpu_example
    COMMAND advanced_evolution_example
    COMMAND advanced_benchmark_example
    DEPENDS advanced_gpu_example advanced_evolution_example advanced_benchmark_example
)

add_custom_target(run_tutorials
    COMMAND echo "Running tutorials..."
    COMMAND tutorial_01_hello_circuit
    COMMAND tutorial_02_circuit_simulation
    COMMAND tutorial_03_genetic_algorithm
    COMMAND tutorial_04_gpu_acceleration
    DEPENDS tutorial_01_hello_circuit tutorial_02_circuit_simulation tutorial_03_genetic_algorithm tutorial_04_gpu_acceleration
)

# Install examples
install(TARGETS 
    basic_circuit_example
    basic_genetic_example
    basic_visualization_example
    advanced_gpu_example
    advanced_evolution_example
    advanced_benchmark_example
    advanced_interactive_example
    tutorial_01_hello_circuit
    tutorial_02_circuit_simulation
    tutorial_03_genetic_algorithm
    tutorial_04_gpu_acceleration
    tutorial_05_visualization
    RUNTIME DESTINATION bin/examples
)

# Copy example data files
install(DIRECTORY data/ DESTINATION share/circuit_designer/examples/data)
install(DIRECTORY configs/ DESTINATION share/circuit_designer/examples/configs) 