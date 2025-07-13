#include <gtest/gtest.h>
#include <chrono>
#include "circuit/ga/genetic_algorithm.h"
#include "circuit/core/types.h"

using namespace circuit;

class GeneticAlgorithmPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid = GridDimensions(16, 16);
        
        params.population_size = 100;
        params.max_generations = 50;
        params.mutation_rate = 0.1f;
        params.crossover_rate = 0.8f;
        params.use_gpu_acceleration = false;
        
        test_cases = {
            TestCase({LogicState::LOW, LogicState::LOW}, {LogicState::LOW}),
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

// Test genetic algorithm performance
TEST_F(GeneticAlgorithmPerformanceTest, BasicEvolutionPerformance) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ASSERT_TRUE(ga != nullptr);
    
    std::mt19937 rng(42);
    
    // Measure evolution time
    auto start = std::chrono::high_resolution_clock::now();
    
    ga->initialize_population(rng, 0.2f);
    ga->evolve(test_cases, fitness_weights, rng);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Evolution took " << duration.count() << " ms" << std::endl;
    
    // Should complete within reasonable time
    EXPECT_LT(duration.count(), 30000);  // Less than 30 seconds
    
    // Check that we got results
    auto best = ga->get_best_genome();
    EXPECT_TRUE(best.is_evaluated());
    EXPECT_GT(best.get_fitness(), 0.0f);
} 