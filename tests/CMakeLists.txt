# Tests CMakeLists.txt

# Enable testing
enable_testing()

# Find required packages for testing
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Download and build Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/release-1.12.1.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Unit tests
add_executable(unit_tests
    unit/test_types.cpp
    unit/test_circuit.cpp
    unit/test_genome.cpp
    unit/test_genetic_algorithm.cpp
    unit/test_gpu_simulator.cpp
    unit/test_visualization.cpp
    unit/test_utils.cpp
    unit/main_test.cpp
)

target_include_directories(unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(unit_tests
    circuit_core
    circuit_ga
    circuit_viz
    circuit_utils
    gtest
    gtest_main
    gmock
    gmock_main
)

# Conditionally link GPU library if available
if(CUDA_FOUND)
    target_link_libraries(unit_tests circuit_gpu)
endif()

# Integration tests
add_executable(integration_tests
    integration/test_end_to_end.cpp
    integration/test_gpu_cpu_comparison.cpp
    integration/test_evolution_scenarios.cpp
    integration/test_visualization_integration.cpp
    integration/main_test.cpp
)

target_include_directories(integration_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(integration_tests
    circuit_core
    circuit_ga
    circuit_viz
    circuit_utils
    gtest
    gtest_main
    gmock
    gmock_main
)

# Conditionally link GPU library if available
if(CUDA_FOUND)
    target_link_libraries(integration_tests circuit_gpu)
endif()

# Performance tests
add_executable(performance_tests
    performance/test_circuit_simulation.cpp
    performance/test_genetic_algorithm.cpp
    performance/test_gpu_performance.cpp
    performance/main_test.cpp
)

target_include_directories(performance_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(performance_tests
    circuit_core
    circuit_ga
    circuit_utils
    gtest
    gtest_main
)

# Conditionally link GPU library if available
if(CUDA_FOUND)
    target_link_libraries(performance_tests circuit_gpu)
endif()

# Register tests with CTest
add_test(NAME UnitTests COMMAND unit_tests)
add_test(NAME IntegrationTests COMMAND integration_tests)
add_test(NAME PerformanceTests COMMAND performance_tests)

# Test configuration
set_tests_properties(UnitTests PROPERTIES TIMEOUT 300)
set_tests_properties(IntegrationTests PROPERTIES TIMEOUT 600)
set_tests_properties(PerformanceTests PROPERTIES TIMEOUT 1200)

# Test data directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Custom targets for different test types
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -R "UnitTests" --verbose
    DEPENDS unit_tests
)

add_custom_target(run_integration_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -R "IntegrationTests" --verbose
    DEPENDS integration_tests
)

add_custom_target(run_performance_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -R "PerformanceTests" --verbose
    DEPENDS performance_tests
)

add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS unit_tests integration_tests performance_tests
)

# Code coverage (if available)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    find_program(LCOV_PATH lcov)
    find_program(GENHTML_PATH genhtml)
    
    if(LCOV_PATH AND GENHTML_PATH)
        add_custom_target(coverage
            COMMAND ${LCOV_PATH} --directory . --zerocounters
            COMMAND ${CMAKE_CTEST_COMMAND} --verbose
            COMMAND ${LCOV_PATH} --directory . --capture --output-file coverage.info
            COMMAND ${LCOV_PATH} --remove coverage.info '/usr/*' --output-file coverage.info
            COMMAND ${GENHTML_PATH} -o coverage coverage.info
            DEPENDS unit_tests integration_tests
            COMMENT "Generating code coverage report"
        )
    endif()
endif()

# Install test binaries (optional)
install(TARGETS unit_tests integration_tests performance_tests
    RUNTIME DESTINATION bin/tests
    OPTIONAL
) 