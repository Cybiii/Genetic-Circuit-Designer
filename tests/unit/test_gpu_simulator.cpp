#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "circuit/gpu/gpu_simulator.h"
#include "circuit/core/types.h"

using namespace circuit;

class GPUSimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        if (!GPUSimulator::is_cuda_available()) {
            GTEST_SKIP() << "CUDA not available, skipping GPU tests";
        }
        
        grid = GridDimensions(8, 8);
        simulator = std::make_unique<GPUSimulator>();
    }
    
    void TearDown() override {
        simulator.reset();
    }
    
    GridDimensions grid;
    std::unique_ptr<GPUSimulator> simulator;
};

// Test GPU initialization
TEST_F(GPUSimulatorTest, Initialization) {
    EXPECT_TRUE(simulator->initialize());
    EXPECT_TRUE(simulator->is_initialized());
    
    auto device_info = simulator->get_device_info();
    EXPECT_GT(device_info.cuda_cores, 0);
    EXPECT_GT(device_info.global_memory_mb, 0);
    EXPECT_GT(device_info.shared_memory_per_block_kb, 0);
    EXPECT_GT(device_info.max_threads_per_block, 0);
    EXPECT_GT(device_info.compute_capability_major, 0);
}

// Test device selection
TEST_F(GPUSimulatorTest, DeviceSelection) {
    int device_count = simulator->get_device_count();
    EXPECT_GE(device_count, 1);
    
    for (int i = 0; i < device_count; i++) {
        EXPECT_TRUE(simulator->select_device(i));
        EXPECT_EQ(simulator->get_current_device(), i);
    }
}

// Test memory allocation
TEST_F(GPUSimulatorTest, MemoryAllocation) {
    ASSERT_TRUE(simulator->initialize());
    
    size_t size = 1024 * 1024;  // 1MB
    void* device_ptr = simulator->allocate_device_memory(size);
    EXPECT_NE(device_ptr, nullptr);
    
    EXPECT_TRUE(simulator->free_device_memory(device_ptr));
}

// Test memory transfer
TEST_F(GPUSimulatorTest, MemoryTransfer) {
    ASSERT_TRUE(simulator->initialize());
    
    std::vector<int> host_data = {1, 2, 3, 4, 5};
    size_t size = host_data.size() * sizeof(int);
    
    void* device_ptr = simulator->allocate_device_memory(size);
    ASSERT_NE(device_ptr, nullptr);
    
    // Host to device
    EXPECT_TRUE(simulator->copy_to_device(device_ptr, host_data.data(), size));
    
    // Device to host
    std::vector<int> result_data(host_data.size());
    EXPECT_TRUE(simulator->copy_from_device(result_data.data(), device_ptr, size));
    
    EXPECT_EQ(result_data, host_data);
    
    simulator->free_device_memory(device_ptr);
}

// Test circuit batch simulation
TEST_F(GPUSimulatorTest, CircuitBatchSimulation) {
    ASSERT_TRUE(simulator->initialize());
    
    // Create simple test circuits
    std::vector<Circuit> circuits;
    for (int i = 0; i < 10; i++) {
        Circuit circuit(grid, 2, 1);
        circuit.add_gate(GateType::AND, Position(1, 1));
        circuit.add_gate(GateType::INPUT, Position(0, 0));
        circuit.add_gate(GateType::OUTPUT, Position(2, 2));
        circuits.push_back(circuit);
    }
    
    // Test inputs
    std::vector<LogicState> inputs = {LogicState::HIGH, LogicState::LOW};
    
    // Simulate batch
    std::vector<SimulationResult> results;
    EXPECT_TRUE(simulator->simulate_circuit_batch(circuits, inputs, results));
    
    EXPECT_EQ(results.size(), circuits.size());
    for (const auto& result : results) {
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.outputs.size(), 1);
        EXPECT_GE(result.total_delay, 0.0f);
    }
}

// Test fitness evaluation
TEST_F(GPUSimulatorTest, FitnessEvaluation) {
    ASSERT_TRUE(simulator->initialize());
    
    // Create test genomes
    std::vector<Genome> genomes;
    for (int i = 0; i < 5; i++) {
        Genome genome(grid, 2, 1);
        std::mt19937 rng(i);
        genome.initialize_random(rng, 0.2f);
        genomes.push_back(genome);
    }
    
    // Test cases
    std::vector<TestCase> test_cases = {
        TestCase({LogicState::LOW, LogicState::LOW}, {LogicState::LOW}),
        TestCase({LogicState::HIGH, LogicState::HIGH}, {LogicState::HIGH})
    };
    
    // Fitness weights
    FitnessComponents weights;
    weights.correctness_weight = 1.0f;
    weights.delay_weight = 0.1f;
    weights.power_weight = 0.1f;
    weights.area_weight = 0.1f;
    
    // Evaluate fitness
    std::vector<float> fitness_scores;
    EXPECT_TRUE(simulator->evaluate_fitness_batch(genomes, test_cases, weights, fitness_scores));
    
    EXPECT_EQ(fitness_scores.size(), genomes.size());
    for (float score : fitness_scores) {
        EXPECT_GE(score, 0.0f);
        EXPECT_LE(score, 1.0f);
    }
}

// Test genetic operations
TEST_F(GPUSimulatorTest, GeneticOperations) {
    ASSERT_TRUE(simulator->initialize());
    
    // Create population
    std::vector<Genome> population;
    std::mt19937 rng(42);
    
    for (int i = 0; i < 20; i++) {
        Genome genome(grid, 2, 1);
        genome.initialize_random(rng, 0.2f);
        population.push_back(genome);
    }
    
    // Test selection
    std::vector<int> selected_indices;
    EXPECT_TRUE(simulator->tournament_selection(population, 10, 3, selected_indices));
    EXPECT_EQ(selected_indices.size(), 10);
    
    // Test crossover
    std::vector<Genome> offspring;
    EXPECT_TRUE(simulator->crossover_batch(population, selected_indices, CrossoverType::SINGLE_POINT, offspring));
    EXPECT_GT(offspring.size(), 0);
    
    // Test mutation
    EXPECT_TRUE(simulator->mutate_batch(offspring, 0.1f, MutationType::GATE_TYPE));
}

// Test performance monitoring
TEST_F(GPUSimulatorTest, PerformanceMonitoring) {
    ASSERT_TRUE(simulator->initialize());
    
    // Start profiling
    simulator->start_profiling();
    
    // Do some work
    std::vector<int> test_data = {1, 2, 3, 4, 5};
    size_t size = test_data.size() * sizeof(int);
    void* device_ptr = simulator->allocate_device_memory(size);
    simulator->copy_to_device(device_ptr, test_data.data(), size);
    simulator->copy_from_device(test_data.data(), device_ptr, size);
    simulator->free_device_memory(device_ptr);
    
    // Stop profiling
    simulator->stop_profiling();
    
    auto profile = simulator->get_performance_profile();
    EXPECT_GT(profile.total_time_ms, 0.0f);
    EXPECT_GT(profile.memory_transfer_time_ms, 0.0f);
    EXPECT_GT(profile.kernel_execution_time_ms, 0.0f);
}

// Test memory management
TEST_F(GPUSimulatorTest, MemoryManagement) {
    ASSERT_TRUE(simulator->initialize());
    
    auto memory_info = simulator->get_memory_info();
    size_t initial_free = memory_info.free_bytes;
    
    // Allocate memory
    std::vector<void*> allocations;
    for (int i = 0; i < 10; i++) {
        void* ptr = simulator->allocate_device_memory(1024 * 1024);  // 1MB each
        if (ptr) {
            allocations.push_back(ptr);
        }
    }
    
    memory_info = simulator->get_memory_info();
    EXPECT_LT(memory_info.free_bytes, initial_free);
    
    // Free memory
    for (void* ptr : allocations) {
        simulator->free_device_memory(ptr);
    }
    
    memory_info = simulator->get_memory_info();
    EXPECT_EQ(memory_info.free_bytes, initial_free);
}

// Test error handling
TEST_F(GPUSimulatorTest, ErrorHandling) {
    ASSERT_TRUE(simulator->initialize());
    
    // Test invalid memory allocation
    void* ptr = simulator->allocate_device_memory(SIZE_MAX);
    EXPECT_EQ(ptr, nullptr);
    
    // Test invalid device selection
    EXPECT_FALSE(simulator->select_device(999));
    
    // Test operations on uninitialized simulator
    GPUSimulator uninit_sim;
    EXPECT_FALSE(uninit_sim.is_initialized());
    EXPECT_EQ(uninit_sim.allocate_device_memory(1024), nullptr);
}

// Test thread safety
TEST_F(GPUSimulatorTest, ThreadSafety) {
    ASSERT_TRUE(simulator->initialize());
    
    // Test that simulator can be safely used from multiple threads
    std::vector<std::thread> threads;
    std::atomic<int> success_count(0);
    
    for (int i = 0; i < 4; i++) {
        threads.emplace_back([this, &success_count]() {
            std::vector<int> data = {1, 2, 3, 4, 5};
            size_t size = data.size() * sizeof(int);
            
            void* ptr = simulator->allocate_device_memory(size);
            if (ptr) {
                if (simulator->copy_to_device(ptr, data.data(), size)) {
                    if (simulator->copy_from_device(data.data(), ptr, size)) {
                        success_count++;
                    }
                }
                simulator->free_device_memory(ptr);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(success_count.load(), 4);
}

// Test cleanup
TEST_F(GPUSimulatorTest, Cleanup) {
    ASSERT_TRUE(simulator->initialize());
    
    // Allocate some memory
    void* ptr = simulator->allocate_device_memory(1024);
    ASSERT_NE(ptr, nullptr);
    
    // Cleanup should free all allocated memory
    simulator->cleanup();
    EXPECT_FALSE(simulator->is_initialized());
}

// Test kernel launch parameters
TEST_F(GPUSimulatorTest, KernelLaunchParameters) {
    ASSERT_TRUE(simulator->initialize());
    
    // Test optimal launch parameters calculation
    auto params = simulator->calculate_optimal_launch_params(1000, 256);
    EXPECT_GT(params.grid_size, 0);
    EXPECT_GT(params.block_size, 0);
    EXPECT_LE(params.block_size, 1024);  // Maximum block size
    
    // Test with different problem sizes
    auto params2 = simulator->calculate_optimal_launch_params(100000, 512);
    EXPECT_GT(params2.grid_size, params.grid_size);
}

// Test CUDA stream management
TEST_F(GPUSimulatorTest, StreamManagement) {
    ASSERT_TRUE(simulator->initialize());
    
    // Create streams
    auto stream1 = simulator->create_stream();
    auto stream2 = simulator->create_stream();
    
    EXPECT_NE(stream1, nullptr);
    EXPECT_NE(stream2, nullptr);
    EXPECT_NE(stream1, stream2);
    
    // Synchronize streams
    EXPECT_TRUE(simulator->synchronize_stream(stream1));
    EXPECT_TRUE(simulator->synchronize_stream(stream2));
    
    // Destroy streams
    EXPECT_TRUE(simulator->destroy_stream(stream1));
    EXPECT_TRUE(simulator->destroy_stream(stream2));
}

// Test specific CUDA features
TEST_F(GPUSimulatorTest, CUDAFeatures) {
    ASSERT_TRUE(simulator->initialize());
    
    auto device_info = simulator->get_device_info();
    
    // Test warp size
    EXPECT_EQ(device_info.warp_size, 32);
    
    // Test shared memory configuration
    if (device_info.compute_capability_major >= 2) {
        EXPECT_TRUE(simulator->configure_shared_memory(SharedMemoryConfig::PREFER_L1));
        EXPECT_TRUE(simulator->configure_shared_memory(SharedMemoryConfig::PREFER_SHARED));
    }
    
    // Test cache preferences
    if (device_info.compute_capability_major >= 2) {
        EXPECT_TRUE(simulator->set_cache_preference(CachePreference::PREFER_L1));
        EXPECT_TRUE(simulator->set_cache_preference(CachePreference::PREFER_SHARED));
    }
} 