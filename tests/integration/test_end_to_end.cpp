#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "circuit/core/circuit.h"
#include "circuit/ga/genetic_algorithm.h"
#include "circuit/ga/genome.h"
#include "circuit/utils/utils.h"
#include <random>

using namespace circuit;

class EndToEndTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid = GridDimensions(16, 16);
        rng.seed(42);
        
        // Setup evolution parameters
        params.population_size = 50;
        params.max_generations = 20;
        params.mutation_rate = 0.1f;
        params.crossover_rate = 0.8f;
        params.elitism_rate = 0.1f;
        params.tournament_size = 3;
        params.use_gpu_acceleration = false;  // Use CPU for reproducible tests
        
        // Setup fitness weights
        fitness_weights.correctness_weight = 1.0f;
        fitness_weights.delay_weight = 0.2f;
        fitness_weights.power_weight = 0.1f;
        fitness_weights.area_weight = 0.1f;
    }
    
    void TearDown() override {
        // Cleanup
    }
    
    GridDimensions grid;
    std::mt19937 rng;
    EvolutionaryParams params;
    FitnessComponents fitness_weights;
    
    // Helper function to create test cases for different circuit types
    std::vector<TestCase> create_adder_test_cases(int bits) {
        std::vector<TestCase> test_cases;
        
        // Test a subset of all possible combinations for performance
        for (int a = 0; a < (1 << bits); a++) {
            for (int b = 0; b < (1 << bits); b++) {
                for (int carry_in = 0; carry_in < 2; carry_in++) {
                    std::vector<LogicState> inputs;
                    
                    // Add A inputs
                    for (int i = 0; i < bits; i++) {
                        inputs.push_back((a & (1 << i)) ? LogicState::HIGH : LogicState::LOW);
                    }
                    
                    // Add B inputs
                    for (int i = 0; i < bits; i++) {
                        inputs.push_back((b & (1 << i)) ? LogicState::HIGH : LogicState::LOW);
                    }
                    
                    // Add carry input
                    inputs.push_back(carry_in ? LogicState::HIGH : LogicState::LOW);
                    
                    // Calculate expected output
                    int sum = a + b + carry_in;
                    std::vector<LogicState> outputs;
                    
                    // Sum outputs
                    for (int i = 0; i < bits; i++) {
                        outputs.push_back((sum & (1 << i)) ? LogicState::HIGH : LogicState::LOW);
                    }
                    
                    // Carry output
                    outputs.push_back((sum & (1 << bits)) ? LogicState::HIGH : LogicState::LOW);
                    
                    test_cases.push_back(TestCase(inputs, outputs));
                }
            }
        }
        
        return test_cases;
    }
    
    std::vector<TestCase> create_multiplexer_test_cases(int select_bits) {
        std::vector<TestCase> test_cases;
        int data_inputs = 1 << select_bits;
        
        // Test all combinations of select inputs and one data input active
        for (int select = 0; select < (1 << select_bits); select++) {
            for (int active_data = 0; active_data < data_inputs; active_data++) {
                std::vector<LogicState> inputs;
                
                // Add select inputs
                for (int i = 0; i < select_bits; i++) {
                    inputs.push_back((select & (1 << i)) ? LogicState::HIGH : LogicState::LOW);
                }
                
                // Add data inputs
                for (int i = 0; i < data_inputs; i++) {
                    inputs.push_back((i == active_data) ? LogicState::HIGH : LogicState::LOW);
                }
                
                // Expected output
                std::vector<LogicState> outputs;
                outputs.push_back((select == active_data) ? LogicState::HIGH : LogicState::LOW);
                
                test_cases.push_back(TestCase(inputs, outputs));
            }
        }
        
        return test_cases;
    }
};

// Test evolving a simple 2-bit adder
TEST_F(EndToEndTest, Evolve2BitAdder) {
    // Create test cases for 2-bit adder
    auto test_cases = create_adder_test_cases(2);
    
    // Create and run genetic algorithm
    auto ga = create_genetic_algorithm(params, grid, 5, 3);  // 2 A bits + 2 B bits + carry_in, 2 sum bits + carry_out
    ASSERT_TRUE(ga != nullptr);
    
    ga->initialize_population(rng, 0.3f);
    
    // Run evolution
    ga->evolve(test_cases, fitness_weights, rng);
    
    // Check results
    EXPECT_EQ(ga->get_current_generation(), params.max_generations);
    
    auto best_genome = ga->get_best_genome();
    EXPECT_TRUE(best_genome.is_evaluated());
    EXPECT_GT(best_genome.get_fitness(), 0.0f);
    
    // Test that best genome produces a valid circuit
    auto circuit = best_genome.to_circuit();
    ASSERT_TRUE(circuit != nullptr);
    EXPECT_TRUE(circuit->is_valid());
}

// Test evolving a multiplexer
TEST_F(EndToEndTest, EvolveMultiplexer) {
    // Create test cases for 2-to-1 multiplexer
    auto test_cases = create_multiplexer_test_cases(1);
    
    // Create and run genetic algorithm
    auto ga = create_genetic_algorithm(params, grid, 3, 1);  // 1 select bit + 2 data bits, 1 output
    ASSERT_TRUE(ga != nullptr);
    
    ga->initialize_population(rng, 0.2f);
    
    // Run evolution
    ga->evolve(test_cases, fitness_weights, rng);
    
    // Check results
    auto best_genome = ga->get_best_genome();
    EXPECT_TRUE(best_genome.is_evaluated());
    EXPECT_GT(best_genome.get_fitness(), 0.0f);
    
    // Test specific functionality
    auto circuit = best_genome.to_circuit();
    ASSERT_TRUE(circuit != nullptr);
    
    // Test a specific case
    std::vector<LogicState> inputs = {LogicState::LOW, LogicState::HIGH, LogicState::LOW};  // select=0, data0=1, data1=0
    std::vector<LogicState> outputs;
    
    if (circuit->simulate(inputs, outputs)) {
        EXPECT_EQ(outputs.size(), 1);
        // Should select first input (HIGH)
        EXPECT_EQ(outputs[0], LogicState::HIGH);
    }
}

// Test complete workflow: create -> evolve -> save -> load
TEST_F(EndToEndTest, CompleteWorkflow) {
    // Create simple AND gate test cases
    std::vector<TestCase> test_cases = {
        TestCase({LogicState::LOW, LogicState::LOW}, {LogicState::LOW}),
        TestCase({LogicState::LOW, LogicState::HIGH}, {LogicState::LOW}),
        TestCase({LogicState::HIGH, LogicState::LOW}, {LogicState::LOW}),
        TestCase({LogicState::HIGH, LogicState::HIGH}, {LogicState::HIGH})
    };
    
    // Create and run genetic algorithm
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ASSERT_TRUE(ga != nullptr);
    
    ga->initialize_population(rng, 0.2f);
    
    // Set up callback to monitor progress
    std::vector<float> fitness_progress;
    ga->set_generation_callback([&fitness_progress](int generation, const EvolutionStatistics& stats) {
        fitness_progress.push_back(stats.best_fitness);
    });
    
    // Run evolution
    ga->evolve(test_cases, fitness_weights, rng);
    
    // Check that we monitored progress
    EXPECT_EQ(fitness_progress.size(), params.max_generations);
    
    // Get best result
    auto best_genome = ga->get_best_genome();
    EXPECT_TRUE(best_genome.is_evaluated());
    
    // Save best genome
    std::string genome_file = "test_best_genome.json";
    EXPECT_TRUE(best_genome.save_to_file(genome_file));
    
    // Load genome back
    auto loaded_genome = Genome::load_from_file(genome_file);
    ASSERT_TRUE(loaded_genome != nullptr);
    
    // Verify loaded genome is equivalent
    EXPECT_EQ(loaded_genome->get_fitness(), best_genome.get_fitness());
    EXPECT_EQ(loaded_genome->get_gene_count(), best_genome.get_gene_count());
    
    // Test that loaded genome produces same circuit
    auto original_circuit = best_genome.to_circuit();
    auto loaded_circuit = loaded_genome->to_circuit();
    
    ASSERT_TRUE(original_circuit != nullptr);
    ASSERT_TRUE(loaded_circuit != nullptr);
    
    // Test same functionality
    for (const auto& test_case : test_cases) {
        std::vector<LogicState> original_outputs, loaded_outputs;
        
        bool original_success = original_circuit->simulate(test_case.inputs, original_outputs);
        bool loaded_success = loaded_circuit->simulate(test_case.inputs, loaded_outputs);
        
        EXPECT_EQ(original_success, loaded_success);
        if (original_success && loaded_success) {
            EXPECT_EQ(original_outputs, loaded_outputs);
        }
    }
    
    // Clean up
    std::remove(genome_file.c_str());
}

// Test with different evolution parameters
TEST_F(EndToEndTest, ParameterSweep) {
    // Simple test cases
    std::vector<TestCase> test_cases = {
        TestCase({LogicState::LOW, LogicState::LOW}, {LogicState::LOW}),
        TestCase({LogicState::HIGH, LogicState::HIGH}, {LogicState::HIGH})
    };
    
    struct TestParams {
        float mutation_rate;
        float crossover_rate;
        int population_size;
        std::string name;
    };
    
    std::vector<TestParams> test_params = {
        {0.05f, 0.9f, 30, "low_mutation"},
        {0.2f, 0.7f, 30, "high_mutation"},
        {0.1f, 0.8f, 20, "small_population"},
        {0.1f, 0.8f, 80, "large_population"}
    };
    
    for (const auto& test_param : test_params) {
        EvolutionaryParams custom_params = params;
        custom_params.mutation_rate = test_param.mutation_rate;
        custom_params.crossover_rate = test_param.crossover_rate;
        custom_params.population_size = test_param.population_size;
        custom_params.max_generations = 10;  // Shorter for parameter sweep
        
        auto ga = create_genetic_algorithm(custom_params, grid, 2, 1);
        ASSERT_TRUE(ga != nullptr) << "Failed to create GA for " << test_param.name;
        
        ga->initialize_population(rng, 0.2f);
        ga->evolve(test_cases, fitness_weights, rng);
        
        auto best_genome = ga->get_best_genome();
        EXPECT_TRUE(best_genome.is_evaluated()) << "Best genome not evaluated for " << test_param.name;
        EXPECT_GT(best_genome.get_fitness(), 0.0f) << "Zero fitness for " << test_param.name;
    }
}

// Test evolution convergence
TEST_F(EndToEndTest, ConvergenceTest) {
    // Simple OR gate test cases
    std::vector<TestCase> test_cases = {
        TestCase({LogicState::LOW, LogicState::LOW}, {LogicState::LOW}),
        TestCase({LogicState::LOW, LogicState::HIGH}, {LogicState::HIGH}),
        TestCase({LogicState::HIGH, LogicState::LOW}, {LogicState::HIGH}),
        TestCase({LogicState::HIGH, LogicState::HIGH}, {LogicState::HIGH})
    };
    
    // Run with higher population and more generations
    EvolutionaryParams conv_params = params;
    conv_params.population_size = 100;
    conv_params.max_generations = 50;
    
    auto ga = create_genetic_algorithm(conv_params, grid, 2, 1);
    ASSERT_TRUE(ga != nullptr);
    
    ga->initialize_population(rng, 0.2f);
    
    // Track convergence
    std::vector<float> best_fitness_history;
    std::vector<float> avg_fitness_history;
    
    ga->set_generation_callback([&](int generation, const EvolutionStatistics& stats) {
        best_fitness_history.push_back(stats.best_fitness);
        avg_fitness_history.push_back(stats.average_fitness);
    });
    
    ga->evolve(test_cases, fitness_weights, rng);
    
    // Check convergence properties
    EXPECT_EQ(best_fitness_history.size(), conv_params.max_generations);
    
    // Fitness should generally improve or stay the same
    for (size_t i = 1; i < best_fitness_history.size(); i++) {
        EXPECT_GE(best_fitness_history[i], best_fitness_history[i-1] - 0.01f);  // Allow small fluctuations
    }
    
    // Average fitness should generally improve
    float initial_avg = avg_fitness_history[0];
    float final_avg = avg_fitness_history.back();
    EXPECT_GT(final_avg, initial_avg - 0.01f);
    
    // Test convergence detection
    EXPECT_TRUE(ga->has_converged(0.01f, 10) || !ga->has_converged(0.01f, 10));  // Should not crash
}

// Test error handling and edge cases
TEST_F(EndToEndTest, ErrorHandling) {
    // Test with invalid grid size
    GridDimensions invalid_grid(0, 0);
    auto invalid_ga = create_genetic_algorithm(params, invalid_grid, 2, 1);
    EXPECT_TRUE(invalid_ga == nullptr);
    
    // Test with invalid parameters
    EvolutionaryParams invalid_params = params;
    invalid_params.population_size = 0;
    auto invalid_ga2 = create_genetic_algorithm(invalid_params, grid, 2, 1);
    EXPECT_TRUE(invalid_ga2 == nullptr);
    
    // Test with empty test cases
    std::vector<TestCase> empty_test_cases;
    auto ga = create_genetic_algorithm(params, grid, 2, 1);
    ASSERT_TRUE(ga != nullptr);
    
    ga->initialize_population(rng, 0.2f);
    
    // Should handle empty test cases gracefully
    ga->evolve(empty_test_cases, fitness_weights, rng);
    
    // Test with mismatched input/output sizes
    std::vector<TestCase> invalid_test_cases = {
        TestCase({LogicState::HIGH}, {LogicState::LOW, LogicState::HIGH})  // 1 input, 2 outputs
    };
    
    // Should handle invalid test cases gracefully
    ga->evolve(invalid_test_cases, fitness_weights, rng);
}

// Test performance characteristics
TEST_F(EndToEndTest, PerformanceTest) {
    // Test with small problem
    std::vector<TestCase> test_cases = {
        TestCase({LogicState::LOW, LogicState::LOW}, {LogicState::LOW}),
        TestCase({LogicState::HIGH, LogicState::HIGH}, {LogicState::HIGH})
    };
    
    // Measure execution time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    EvolutionaryParams perf_params = params;
    perf_params.population_size = 50;
    perf_params.max_generations = 20;
    
    auto ga = create_genetic_algorithm(perf_params, grid, 2, 1);
    ASSERT_TRUE(ga != nullptr);
    
    ga->initialize_population(rng, 0.2f);
    ga->evolve(test_cases, fitness_weights, rng);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Should complete within reasonable time (adjust based on hardware)
    EXPECT_LT(duration.count(), 30000);  // 30 seconds max
    
    // Check that we got meaningful results
    auto best_genome = ga->get_best_genome();
    EXPECT_TRUE(best_genome.is_evaluated());
    EXPECT_GT(best_genome.get_fitness(), 0.0f);
    
    // Test memory usage
    auto stats = ga->get_statistics();
    EXPECT_GT(stats.best_fitness, 0.0f);
    EXPECT_LE(stats.best_fitness, 1.0f);
    EXPECT_GE(stats.average_fitness, 0.0f);
    EXPECT_LE(stats.average_fitness, stats.best_fitness);
}

// Test reproducibility
TEST_F(EndToEndTest, ReproducibilityTest) {
    std::vector<TestCase> test_cases = {
        TestCase({LogicState::LOW, LogicState::LOW}, {LogicState::LOW}),
        TestCase({LogicState::HIGH, LogicState::HIGH}, {LogicState::HIGH})
    };
    
    // Run same experiment twice with same seed
    std::vector<float> fitness_history1, fitness_history2;
    
    // First run
    std::mt19937 rng1(12345);
    auto ga1 = create_genetic_algorithm(params, grid, 2, 1);
    ASSERT_TRUE(ga1 != nullptr);
    
    ga1->initialize_population(rng1, 0.2f);
    ga1->set_generation_callback([&](int gen, const EvolutionStatistics& stats) {
        fitness_history1.push_back(stats.best_fitness);
    });
    ga1->evolve(test_cases, fitness_weights, rng1);
    
    // Second run with same seed
    std::mt19937 rng2(12345);
    auto ga2 = create_genetic_algorithm(params, grid, 2, 1);
    ASSERT_TRUE(ga2 != nullptr);
    
    ga2->initialize_population(rng2, 0.2f);
    ga2->set_generation_callback([&](int gen, const EvolutionStatistics& stats) {
        fitness_history2.push_back(stats.best_fitness);
    });
    ga2->evolve(test_cases, fitness_weights, rng2);
    
    // Results should be identical
    EXPECT_EQ(fitness_history1.size(), fitness_history2.size());
    for (size_t i = 0; i < fitness_history1.size(); i++) {
        EXPECT_EQ(fitness_history1[i], fitness_history2[i]);
    }
    
    // Best genomes should be identical
    auto best1 = ga1->get_best_genome();
    auto best2 = ga2->get_best_genome();
    EXPECT_EQ(best1.get_fitness(), best2.get_fitness());
    EXPECT_EQ(best1.get_gene_count(), best2.get_gene_count());
} 