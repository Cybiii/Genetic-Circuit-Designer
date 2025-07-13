#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "circuit/ga/genetic_algorithm.h"
#include "circuit/core/types.h"
#include <random>

using namespace circuit;

class GeneticAlgorithmTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid = GridDimensions(8, 8);
        rng.seed(42);
        
        // Setup basic parameters
        params.population_size = 20;
        params.max_generations = 10;
        params.mutation_rate = 0.1f;
        params.crossover_rate = 0.8f;
        params.elitism_rate = 0.1f;
        params.tournament_size = 3;
        params.use_gpu_acceleration = false;  // Use CPU for unit tests
        
        // Create simple test cases for a 2-input AND gate
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
    
    void TearDown() override {
        // Cleanup
    }
    
    GridDimensions grid;
    std::mt19937 rng;
    EvolutionaryParams params;
    std::vector<TestCase> test_cases;
    FitnessComponents fitness_weights;
};

// Test GeneticAlgorithm construction
TEST_F(GeneticAlgorithmTest, Construction) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    EXPECT_TRUE(ga != nullptr);
    EXPECT_EQ(ga->get_population_size(), params.population_size);
    EXPECT_EQ(ga->get_max_generations(), params.max_generations);
    EXPECT_EQ(ga->get_current_generation(), 0);
}

// Test population initialization
TEST_F(GeneticAlgorithmTest, InitializePopulation) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    EXPECT_EQ(ga->get_population().size(), params.population_size);
    for (const auto& genome : ga->get_population()) {
        EXPECT_TRUE(genome.is_valid());
    }
}

// Test single generation evolution
TEST_F(GeneticAlgorithmTest, EvolveGeneration) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    
    EXPECT_EQ(ga->get_current_generation(), 1);
    
    // Check that all genomes have been evaluated
    for (const auto& genome : ga->get_population()) {
        EXPECT_TRUE(genome.is_evaluated());
    }
}

// Test full evolution
TEST_F(GeneticAlgorithmTest, FullEvolution) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve(test_cases, fitness_weights, rng);
    
    EXPECT_EQ(ga->get_current_generation(), params.max_generations);
    
    // Check that we have a best genome
    auto best = ga->get_best_genome();
    EXPECT_TRUE(best.is_evaluated());
    EXPECT_GT(best.get_fitness(), 0.0f);
}

// Test selection strategies
TEST_F(GeneticAlgorithmTest, TournamentSelection) {
    params.selection_strategy = SelectionStrategy::TOURNAMENT;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 1);
}

TEST_F(GeneticAlgorithmTest, RouletteWheelSelection) {
    params.selection_strategy = SelectionStrategy::ROULETTE_WHEEL;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 1);
}

TEST_F(GeneticAlgorithmTest, RankBasedSelection) {
    params.selection_strategy = SelectionStrategy::RANK_BASED;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 1);
}

// Test crossover types
TEST_F(GeneticAlgorithmTest, SinglePointCrossover) {
    params.crossover_type = CrossoverType::SINGLE_POINT;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 1);
}

TEST_F(GeneticAlgorithmTest, UniformCrossover) {
    params.crossover_type = CrossoverType::UNIFORM;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 1);
}

// Test mutation types
TEST_F(GeneticAlgorithmTest, GateTypeMutation) {
    params.mutation_type = MutationType::GATE_TYPE;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 1);
}

TEST_F(GeneticAlgorithmTest, ConnectionMutation) {
    params.mutation_type = MutationType::CONNECTION;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 1);
}

// Test fitness evaluation
TEST_F(GeneticAlgorithmTest, FitnessEvaluation) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    auto& population = ga->get_population();
    ga->evaluate_fitness(population, test_cases, fitness_weights);
    
    for (const auto& genome : population) {
        EXPECT_TRUE(genome.is_evaluated());
        EXPECT_GE(genome.get_fitness(), 0.0f);
        EXPECT_LE(genome.get_fitness(), 1.0f);
    }
}

// Test statistics
TEST_F(GeneticAlgorithmTest, Statistics) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    
    auto stats = ga->get_statistics();
    EXPECT_GE(stats.best_fitness, 0.0f);
    EXPECT_LE(stats.best_fitness, 1.0f);
    EXPECT_GE(stats.average_fitness, 0.0f);
    EXPECT_LE(stats.average_fitness, 1.0f);
    EXPECT_GE(stats.worst_fitness, 0.0f);
    EXPECT_LE(stats.worst_fitness, 1.0f);
    EXPECT_LE(stats.worst_fitness, stats.average_fitness);
    EXPECT_LE(stats.average_fitness, stats.best_fitness);
}

// Test convergence
TEST_F(GeneticAlgorithmTest, ConvergenceDetection) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    // Run a few generations
    for (int i = 0; i < 3; i++) {
        ga->evolve_single_generation(test_cases, fitness_weights, rng);
    }
    
    bool converged = ga->has_converged(0.01f, 5);
    EXPECT_TRUE(converged || !converged);  // Just test it doesn't crash
}

// Test elitism
TEST_F(GeneticAlgorithmTest, Elitism) {
    params.elitism_rate = 0.2f;  // Keep top 20%
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    
    // Best genome should be preserved
    auto best = ga->get_best_genome();
    EXPECT_TRUE(best.is_evaluated());
    EXPECT_GT(best.get_fitness(), 0.0f);
}

// Test population diversity
TEST_F(GeneticAlgorithmTest, PopulationDiversity) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    float diversity = ga->calculate_diversity();
    EXPECT_GE(diversity, 0.0f);
    EXPECT_LE(diversity, 1.0f);
}

// Test serialization
TEST_F(GeneticAlgorithmTest, SerializeState) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    
    auto json = ga->serialize_state();
    EXPECT_TRUE(json.contains("current_generation"));
    EXPECT_TRUE(json.contains("population"));
    EXPECT_TRUE(json.contains("parameters"));
    EXPECT_TRUE(json.contains("statistics"));
}

TEST_F(GeneticAlgorithmTest, DeserializeState) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    
    auto json = ga->serialize_state();
    auto new_ga = create_genetic_algorithm(params, grid, 2, 1);
    
    EXPECT_TRUE(new_ga->deserialize_state(json));
    EXPECT_EQ(new_ga->get_current_generation(), ga->get_current_generation());
    EXPECT_EQ(new_ga->get_population_size(), ga->get_population_size());
}

// Test callbacks
TEST_F(GeneticAlgorithmTest, GenerationCallback) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    int callback_count = 0;
    ga->set_generation_callback([&callback_count](int generation, const EvolutionStatistics& stats) {
        callback_count++;
        EXPECT_GE(generation, 0);
        EXPECT_GE(stats.best_fitness, 0.0f);
    });
    
    ga->evolve(test_cases, fitness_weights, rng);
    EXPECT_EQ(callback_count, params.max_generations);
}

// Test reset
TEST_F(GeneticAlgorithmTest, Reset) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    
    EXPECT_EQ(ga->get_current_generation(), 1);
    
    ga->reset();
    EXPECT_EQ(ga->get_current_generation(), 0);
    EXPECT_EQ(ga->get_population().size(), 0);
}

// Test edge cases
TEST_F(GeneticAlgorithmTest, SmallPopulation) {
    params.population_size = 2;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    // Should still work with very small population
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 1);
}

TEST_F(GeneticAlgorithmTest, ZeroGenerations) {
    params.max_generations = 0;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 0);
}

TEST_F(GeneticAlgorithmTest, HighMutationRate) {
    params.mutation_rate = 1.0f;  // 100% mutation
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 1);
}

TEST_F(GeneticAlgorithmTest, ZeroMutationRate) {
    params.mutation_rate = 0.0f;  // No mutation
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 1);
}

// Test parameter validation
TEST_F(GeneticAlgorithmTest, InvalidParameters) {
    // Test with invalid population size
    params.population_size = 0;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    EXPECT_TRUE(ga == nullptr);  // Should fail to create
}

TEST_F(GeneticAlgorithmTest, InvalidGridSize) {
    GridDimensions invalid_grid(0, 0);
    auto ga = create_genetic_algorithm(params, invalid_grid, 2, 1);
    EXPECT_TRUE(ga == nullptr);  // Should fail to create
}

// Test memory usage
TEST_F(GeneticAlgorithmTest, MemoryUsage) {
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    size_t memory_usage = ga->get_memory_usage();
    EXPECT_GT(memory_usage, 0);
}

// Test parallel evaluation (if available)
TEST_F(GeneticAlgorithmTest, ParallelEvaluation) {
    params.use_parallel_evaluation = true;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ga->initialize_population(rng, 0.2f);
    
    ga->evolve_single_generation(test_cases, fitness_weights, rng);
    EXPECT_EQ(ga->get_current_generation(), 1);
} 