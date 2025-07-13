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
        // Convert LogicState inputs to SignalValue
        std::vector<SignalValue> signal_inputs;
        for (const auto& input : test_case.inputs) {
            signal_inputs.push_back(logic_state_to_signal(input));
        }
        
        // Simulate circuit with test case
        std::vector<SignalValue> outputs;
        bool success = circuit->simulate(signal_inputs, outputs);
        
        // Check correctness
        bool correct = true;
        if (success && outputs.size() == test_case.expected_outputs.size()) {
            for (uint32_t i = 0; i < outputs.size(); ++i) {
                LogicState expected = test_case.expected_outputs[i];
                LogicState actual = signal_to_logic_state(outputs[i]);
                if (actual != expected) {
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
        
        // Get performance metrics from circuit
        auto performance = circuit->evaluate_performance(std::vector<TestCase>{test_case});
        total_delay += performance.total_delay;
        total_power += performance.power_consumption;
        total_area += performance.area_cost;
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
                // Convert LogicState inputs to SignalValue
                std::vector<SignalValue> signal_inputs;
                for (const auto& input : test_case.inputs) {
                    signal_inputs.push_back(logic_state_to_signal(input));
                }
                
                std::vector<SignalValue> outputs;
                bool success = circuit->simulate(signal_inputs, outputs);
                
                // Check correctness
                bool correct = true;
                if (success && outputs.size() == test_case.expected_outputs.size()) {
                    for (uint32_t j = 0; j < outputs.size(); ++j) {
                        LogicState expected = test_case.expected_outputs[j];
                        LogicState actual = signal_to_logic_state(outputs[j]);
                        if (actual != expected) {
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
                
                auto performance = circuit->evaluate_performance(std::vector<TestCase>{test_case});
                total_delay += performance.total_delay;
                total_power += performance.power_consumption;
                total_area += performance.area_cost;
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
        // Convert LogicState inputs to SignalValue
        std::vector<SignalValue> signal_inputs;
        for (const auto& input : test_case.inputs) {
            signal_inputs.push_back(logic_state_to_signal(input));
        }
        
        std::vector<SignalValue> outputs;
        bool success = circuit->simulate(signal_inputs, outputs);
        
        // Check correctness
        bool correct = true;
        if (success && outputs.size() == test_case.expected_outputs.size()) {
            for (uint32_t i = 0; i < outputs.size(); ++i) {
                LogicState expected = test_case.expected_outputs[i];
                LogicState actual = signal_to_logic_state(outputs[i]);
                if (actual != expected) {
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
        
        auto performance = circuit->evaluate_performance(std::vector<TestCase>{test_case});
        total_delay += performance.total_delay;
        total_power += performance.power_consumption;
        total_area += performance.area_cost;
    }
    
    // Normalize and return objectives
    objectives.push_back(total_correctness / test_cases.size());  // Maximize correctness
    objectives.push_back(-total_delay / test_cases.size());       // Minimize delay (negative for maximization)
    objectives.push_back(-total_power / test_cases.size());       // Minimize power
    objectives.push_back(-total_area / test_cases.size());        // Minimize area
    
    return objectives;
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