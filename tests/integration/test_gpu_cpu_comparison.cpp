#include <gtest/gtest.h>
#include <chrono>
#include "circuit/ga/genetic_algorithm.h"
#include "circuit/core/types.h"

using namespace circuit;

class GPUCPUComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid = GridDimensions(16, 16);
        
        params.population_size = 50;
        params.max_generations = 10;
        params.mutation_rate = 0.1f;
        params.crossover_rate = 0.8f;
        params.elitism_rate = 0.1f;
        params.tournament_size = 3;
        
        test_cases = {
            TestCase({LogicState::LOW, LogicState::LOW}, {LogicState::LOW}),
            TestCase({LogicState::LOW, LogicState::HIGH}, {LogicState::LOW}),
            TestCase({LogicState::HIGH, LogicState::LOW}, {LogicState::LOW}),
            TestCase({LogicState::HIGH, LogicState::HIGH}, {LogicState::HIGH})
        };
        
        fitness_weights.correctness_weight = 1.0f;
        fitness_weights.delay_weight = 0.1f;
        fitness_weights.power_weight = 0.1f;
        fitness_weights.area_weight = 0.1f;
    }
    
    GridDimensions grid;
    EvolutionaryParams params;
    std::vector<TestCase> test_cases;
    FitnessComponents fitness_weights;
};

// Test CPU vs GPU evolution performance
TEST_F(GPUCPUComparisonTest, PerformanceComparison) {
    std::mt19937 rng_cpu(42);
    std::mt19937 rng_gpu(42);
    
    // Test CPU evolution
    params.use_gpu_acceleration = false;
    auto cpu_ga = create_genetic_algorithm(params, grid, 2, 1);
    ASSERT_TRUE(cpu_ga != nullptr);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    cpu_ga->initialize_population(rng_cpu, 0.2f);
    cpu_ga->evolve(test_cases, fitness_weights, rng_cpu);
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    
    auto cpu_best = cpu_ga->get_best_genome();
    
    // Test GPU evolution (if available)
    if (GPUSimulator::is_cuda_available()) {
        params.use_gpu_acceleration = true;
        auto gpu_ga = create_genetic_algorithm(params, grid, 2, 1);
        ASSERT_TRUE(gpu_ga != nullptr);
        
        auto gpu_start = std::chrono::high_resolution_clock::now();
        
        gpu_ga->initialize_population(rng_gpu, 0.2f);
        gpu_ga->evolve(test_cases, fitness_weights, rng_gpu);
        
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
        
        auto gpu_best = gpu_ga->get_best_genome();
        
        // Compare results
        std::cout << "CPU Evolution Time: " << cpu_duration.count() << " ms" << std::endl;
        std::cout << "GPU Evolution Time: " << gpu_duration.count() << " ms" << std::endl;
        std::cout << "CPU Best Fitness: " << cpu_best.get_fitness() << std::endl;
        std::cout << "GPU Best Fitness: " << gpu_best.get_fitness() << std::endl;
        
        // Both should produce valid results
        EXPECT_TRUE(cpu_best.is_evaluated());
        EXPECT_TRUE(gpu_best.is_evaluated());
        EXPECT_GT(cpu_best.get_fitness(), 0.0f);
        EXPECT_GT(gpu_best.get_fitness(), 0.0f);
        
        // Results should be similar in quality (within tolerance)
        float fitness_diff = std::abs(cpu_best.get_fitness() - gpu_best.get_fitness());
        EXPECT_LT(fitness_diff, 0.2f);  // Allow some variation
    } else {
        std::cout << "GPU not available, skipping GPU comparison" << std::endl;
        std::cout << "CPU Evolution Time: " << cpu_duration.count() << " ms" << std::endl;
        std::cout << "CPU Best Fitness: " << cpu_best.get_fitness() << std::endl;
        
        // Just verify CPU results are valid
        EXPECT_TRUE(cpu_best.is_evaluated());
        EXPECT_GT(cpu_best.get_fitness(), 0.0f);
    }
} 