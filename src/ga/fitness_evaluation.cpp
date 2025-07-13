#include "circuit/ga/genetic_algorithm.h"
#include "circuit/core/types.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace circuit {

// Basic fitness evaluation for a single genome
float evaluate_genome_fitness(const Genome& genome, 
                             const std::vector<TestCase>& test_cases,
                             const FitnessComponents& weights) {
    // Convert genome to circuit
    auto circuit = genome.to_circuit();
    if (!circuit) {
        return 0.0f;
    }
    
    // Evaluate circuit performance
    float total_correctness = 0.0f;
    float total_delay = 0.0f;
    float total_power = 0.0f;
    float total_area = 0.0f;
    
    uint32_t valid_tests = 0;
    
    for (const auto& test_case : test_cases) {
        // Simulate circuit with test case
        std::vector<SignalValue> outputs;
        bool success = circuit->simulate(test_case.inputs, outputs);
        
        // Check correctness
        bool correct = true;
        if (success && outputs.size() == test_case.expected_outputs.size()) {
            for (uint32_t i = 0; i < outputs.size(); ++i) {
                if (outputs[i] != test_case.expected_outputs[i]) {
                    correct = false;
                    break;
                }
            }
        } else {
            correct = false;
        }
        
        if (correct) {
            total_correctness += 1.0f;
        }
        
        // Accumulate performance metrics
        total_delay += result.propagation_delay;
        total_power += result.power_consumption;
        total_area += result.area_cost;
        valid_tests++;
    }
    
    if (valid_tests == 0) {
        return 0.0f;
    }
    
    // Normalize metrics
    float correctness = total_correctness / valid_tests;
    float avg_delay = total_delay / valid_tests;
    float avg_power = total_power / valid_tests;
    float avg_area = total_area / valid_tests;
    
    // Calculate composite fitness
    float fitness = weights.correctness_weight * correctness;
    
    // Add penalties for delay, power, and area (lower is better)
    if (avg_delay > 0.0f) {
        fitness -= weights.delay_weight * avg_delay;
    }
    
    if (avg_power > 0.0f) {
        fitness -= weights.power_weight * avg_power;
    }
    
    if (avg_area > 0.0f) {
        fitness -= weights.area_weight * avg_area;
    }
    
    // Ensure fitness is non-negative
    return std::max(0.0f, fitness);
}

// Batch fitness evaluation for population
void evaluate_population_fitness(GenomePopulation& population,
                                const std::vector<TestCase>& test_cases,
                                const FitnessComponents& weights) {
    for (uint32_t i = 0; i < population.size(); ++i) {
        float fitness = evaluate_genome_fitness(population[i], test_cases, weights);
        population[i].set_fitness(fitness);
        
        // Set detailed performance metrics
        auto circuit = population[i].to_circuit();
        if (circuit) {
            PerformanceMetrics metrics;
            float total_delay = 0.0f;
            float total_power = 0.0f;
            float total_area = 0.0f;
            float correctness = 0.0f;
            
            for (const auto& test_case : test_cases) {
                auto result = circuit->simulate(test_case.inputs);
                
                // Check correctness
                bool correct = true;
                if (result.outputs.size() == test_case.expected_outputs.size()) {
                    for (uint32_t j = 0; j < result.outputs.size(); ++j) {
                        if (result.outputs[j] != test_case.expected_outputs[j]) {
                            correct = false;
                            break;
                        }
                    }
                } else {
                    correct = false;
                }
                
                if (correct) {
                    correctness += 1.0f;
                }
                
                total_delay += result.propagation_delay;
                total_power += result.power_consumption;
                total_area += result.area_cost;
            }
            
            metrics.correctness = correctness / test_cases.size();
            metrics.propagation_delay = total_delay / test_cases.size();
            metrics.power_consumption = total_power / test_cases.size();
            metrics.area_cost = total_area / test_cases.size();
            
            population[i].set_performance_metrics(metrics);
        }
    }
}

// Fitness evaluation with constraint handling
float evaluate_constrained_fitness(const Genome& genome,
                                  const std::vector<TestCase>& test_cases,
                                  const FitnessComponents& weights,
                                  const std::vector<std::function<bool(const Genome&)>>& constraints) {
    // Check constraints first
    for (const auto& constraint : constraints) {
        if (!constraint(genome)) {
            return 0.0f; // Heavily penalize constraint violations
        }
    }
    
    // Evaluate normal fitness
    return evaluate_genome_fitness(genome, test_cases, weights);
}

// Multi-objective fitness evaluation
std::vector<float> evaluate_multi_objective_fitness(const Genome& genome,
                                                   const std::vector<TestCase>& test_cases) {
    std::vector<float> objectives;
    
    auto circuit = genome.to_circuit();
    if (!circuit) {
        return {0.0f, 0.0f, 0.0f, 0.0f}; // Return zeros for all objectives
    }
    
    float total_correctness = 0.0f;
    float total_delay = 0.0f;
    float total_power = 0.0f;
    float total_area = 0.0f;
    
    for (const auto& test_case : test_cases) {
        auto result = circuit->simulate(test_case.inputs);
        
        // Check correctness
        bool correct = true;
        if (result.outputs.size() == test_case.expected_outputs.size()) {
            for (uint32_t i = 0; i < result.outputs.size(); ++i) {
                if (result.outputs[i] != test_case.expected_outputs[i]) {
                    correct = false;
                    break;
                }
            }
        } else {
            correct = false;
        }
        
        if (correct) {
            total_correctness += 1.0f;
        }
        
        total_delay += result.propagation_delay;
        total_power += result.power_consumption;
        total_area += result.area_cost;
    }
    
    // Normalize and return objectives
    objectives.push_back(total_correctness / test_cases.size());  // Maximize correctness
    objectives.push_back(-total_delay / test_cases.size());       // Minimize delay (negative for maximization)
    objectives.push_back(-total_power / test_cases.size());       // Minimize power
    objectives.push_back(-total_area / test_cases.size());        // Minimize area
    
    return objectives;
}

// Fitness evaluation with dynamic test cases
float evaluate_dynamic_fitness(const Genome& genome,
                              const std::vector<TestCase>& base_test_cases,
                              const FitnessComponents& weights,
                              uint32_t generation) {
    // Create dynamic test cases based on generation
    std::vector<TestCase> dynamic_tests = base_test_cases;
    
    // Add more challenging test cases as evolution progresses
    if (generation > 100) {
        // Add stress test cases
        for (const auto& base_test : base_test_cases) {
            TestCase stress_test = base_test;
            // Modify inputs to create edge cases
            for (auto& input : stress_test.inputs) {
                input = (input == LogicState::HIGH) ? LogicState::LOW : LogicState::HIGH;
            }
            dynamic_tests.push_back(stress_test);
        }
    }
    
    return evaluate_genome_fitness(genome, dynamic_tests, weights);
}

// Fitness evaluation with noise robustness
float evaluate_robust_fitness(const Genome& genome,
                             const std::vector<TestCase>& test_cases,
                             const FitnessComponents& weights,
                             float noise_level,
                             std::mt19937& rng) {
    auto circuit = genome.to_circuit();
    if (!circuit) {
        return 0.0f;
    }
    
    float total_fitness = 0.0f;
    const uint32_t noise_trials = 10;
    
    std::uniform_real_distribution<float> noise_dist(0.0f, 1.0f);
    
    for (uint32_t trial = 0; trial < noise_trials; ++trial) {
        std::vector<TestCase> noisy_tests;
        
        // Add noise to test cases
        for (const auto& test_case : test_cases) {
            TestCase noisy_test = test_case;
            
            // Add noise to inputs
            for (auto& input : noisy_test.inputs) {
                if (noise_dist(rng) < noise_level) {
                    // Flip the input
                    input = (input == LogicState::HIGH) ? LogicState::LOW : LogicState::HIGH;
                }
            }
            
            noisy_tests.push_back(noisy_test);
        }
        
        // Evaluate with noisy test cases
        float trial_fitness = evaluate_genome_fitness(genome, noisy_tests, weights);
        total_fitness += trial_fitness;
    }
    
    return total_fitness / noise_trials;
}

// Fitness evaluation with complexity penalty
float evaluate_complexity_fitness(const Genome& genome,
                                 const std::vector<TestCase>& test_cases,
                                 const FitnessComponents& weights,
                                 float complexity_penalty) {
    float base_fitness = evaluate_genome_fitness(genome, test_cases, weights);
    
    // Calculate complexity penalty
    auto stats = genome.get_statistics();
    float complexity = stats.complexity_score;
    
    // Apply penalty
    float penalty = complexity_penalty * complexity;
    
    return std::max(0.0f, base_fitness - penalty);
}

// Fitness evaluation with diversity bonus
float evaluate_diversity_fitness(const Genome& genome,
                                const std::vector<TestCase>& test_cases,
                                const FitnessComponents& weights,
                                const GenomePopulation& population,
                                float diversity_bonus) {
    float base_fitness = evaluate_genome_fitness(genome, test_cases, weights);
    
    // Calculate diversity bonus
    float min_similarity = 1.0f;
    for (uint32_t i = 0; i < population.size(); ++i) {
        float similarity = calculate_genome_similarity(genome, population[i]);
        min_similarity = std::min(min_similarity, similarity);
    }
    
    float diversity = 1.0f - min_similarity;
    float bonus = diversity_bonus * diversity;
    
    return base_fitness + bonus;
}

// Fitness evaluation with hierarchical decomposition
float evaluate_hierarchical_fitness(const Genome& genome,
                                   const std::vector<TestCase>& test_cases,
                                   const FitnessComponents& weights) {
    // Break down fitness evaluation into hierarchical components
    
    // Level 1: Basic functionality
    float functionality_score = 0.0f;
    auto circuit = genome.to_circuit();
    if (circuit) {
        uint32_t correct_outputs = 0;
        for (const auto& test_case : test_cases) {
            auto result = circuit->simulate(test_case.inputs);
            if (result.outputs.size() == test_case.expected_outputs.size()) {
                bool all_correct = true;
                for (uint32_t i = 0; i < result.outputs.size(); ++i) {
                    if (result.outputs[i] != test_case.expected_outputs[i]) {
                        all_correct = false;
                        break;
                    }
                }
                if (all_correct) {
                    correct_outputs++;
                }
            }
        }
        functionality_score = static_cast<float>(correct_outputs) / test_cases.size();
    }
    
    // Level 2: Performance metrics (only if functionality is good)
    float performance_score = 0.0f;
    if (functionality_score > 0.5f) {
        float total_delay = 0.0f;
        float total_power = 0.0f;
        
        for (const auto& test_case : test_cases) {
            auto result = circuit->simulate(test_case.inputs);
            total_delay += result.propagation_delay;
            total_power += result.power_consumption;
        }
        
        // Normalize performance (lower is better)
        float avg_delay = total_delay / test_cases.size();
        float avg_power = total_power / test_cases.size();
        
        performance_score = 1.0f / (1.0f + avg_delay + avg_power);
    }
    
    // Level 3: Complexity and elegance (only if performance is good)
    float elegance_score = 0.0f;
    if (performance_score > 0.3f) {
        auto stats = genome.get_statistics();
        float complexity = stats.complexity_score;
        elegance_score = 1.0f / (1.0f + complexity);
    }
    
    // Combine hierarchical scores
    float total_fitness = weights.correctness_weight * functionality_score +
                         weights.delay_weight * performance_score +
                         weights.power_weight * elegance_score;
    
    return total_fitness;
}

// Adaptive fitness evaluation
float evaluate_adaptive_fitness(const Genome& genome,
                               const std::vector<TestCase>& test_cases,
                               FitnessComponents& weights,
                               uint32_t generation) {
    // Adapt weights based on generation
    if (generation < 100) {
        // Early generations: focus on correctness
        weights.correctness_weight = 1.0f;
        weights.delay_weight = 0.1f;
        weights.power_weight = 0.1f;
        weights.area_weight = 0.1f;
    } else if (generation < 300) {
        // Mid generations: balance correctness and performance
        weights.correctness_weight = 0.7f;
        weights.delay_weight = 0.3f;
        weights.power_weight = 0.2f;
        weights.area_weight = 0.2f;
    } else {
        // Late generations: optimize performance
        weights.correctness_weight = 0.5f;
        weights.delay_weight = 0.4f;
        weights.power_weight = 0.3f;
        weights.area_weight = 0.3f;
    }
    
    return evaluate_genome_fitness(genome, test_cases, weights);
}

// Fitness evaluation with partial credit
float evaluate_partial_credit_fitness(const Genome& genome,
                                     const std::vector<TestCase>& test_cases,
                                     const FitnessComponents& weights) {
    auto circuit = genome.to_circuit();
    if (!circuit) {
        return 0.0f;
    }
    
    float total_score = 0.0f;
    
    for (const auto& test_case : test_cases) {
        auto result = circuit->simulate(test_case.inputs);
        
        if (result.outputs.size() == test_case.expected_outputs.size()) {
            // Calculate partial credit for each output
            float output_score = 0.0f;
            for (uint32_t i = 0; i < result.outputs.size(); ++i) {
                if (result.outputs[i] == test_case.expected_outputs[i]) {
                    output_score += 1.0f;
                } else {
                    // Partial credit for "close" outputs
                    output_score += 0.1f;
                }
            }
            
            total_score += output_score / result.outputs.size();
        }
    }
    
    float correctness = total_score / test_cases.size();
    
    // Apply performance penalties
    float total_delay = 0.0f;
    float total_power = 0.0f;
    float total_area = 0.0f;
    
    for (const auto& test_case : test_cases) {
        auto result = circuit->simulate(test_case.inputs);
        total_delay += result.propagation_delay;
        total_power += result.power_consumption;
        total_area += result.area_cost;
    }
    
    float avg_delay = total_delay / test_cases.size();
    float avg_power = total_power / test_cases.size();
    float avg_area = total_area / test_cases.size();
    
    float fitness = weights.correctness_weight * correctness -
                   weights.delay_weight * avg_delay -
                   weights.power_weight * avg_power -
                   weights.area_weight * avg_area;
    
    return std::max(0.0f, fitness);
}

// Utility function to create fitness components for different problems
FitnessComponents create_fitness_components_for_problem(const std::string& problem_type) {
    FitnessComponents components;
    
    if (problem_type == "adder") {
        components.correctness_weight = 1.0f;
        components.delay_weight = 0.3f;
        components.power_weight = 0.2f;
        components.area_weight = 0.1f;
    } else if (problem_type == "multiplexer") {
        components.correctness_weight = 1.0f;
        components.delay_weight = 0.4f;
        components.power_weight = 0.1f;
        components.area_weight = 0.1f;
    } else if (problem_type == "comparator") {
        components.correctness_weight = 1.0f;
        components.delay_weight = 0.2f;
        components.power_weight = 0.3f;
        components.area_weight = 0.2f;
    } else {
        // Default balanced weights
        components.correctness_weight = 1.0f;
        components.delay_weight = 0.25f;
        components.power_weight = 0.25f;
        components.area_weight = 0.25f;
    }
    
    return components;
}

} // namespace circuit 