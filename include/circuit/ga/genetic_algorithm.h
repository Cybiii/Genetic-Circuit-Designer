#pragma once

#include "../core/types.h"
#include "../core/circuit.h"
#include "genome.h"
#include <vector>
#include <memory>
#include <functional>
#include <random>
#include <chrono>

namespace circuit {

// Forward declarations
class GPUSimulator;

// Selection strategies
enum class SelectionStrategy {
    TOURNAMENT,
    ROULETTE_WHEEL,
    RANK_BASED,
    ELITIST
};

// Evolutionary parameters
struct EvolutionaryParams {
    uint32_t population_size;
    uint32_t max_generations;
    float mutation_rate;
    float crossover_rate;
    uint32_t tournament_size;
    uint32_t elite_count;
    SelectionStrategy selection_strategy;
    CrossoverType crossover_type;
    bool use_gpu_acceleration;
    
    EvolutionaryParams() : population_size(100), max_generations(500),
                          mutation_rate(0.1f), crossover_rate(0.8f),
                          tournament_size(4), elite_count(5),
                          selection_strategy(SelectionStrategy::TOURNAMENT),
                          crossover_type(CrossoverType::TWO_POINT),
                          use_gpu_acceleration(true) {}
};

// Evolution statistics
struct EvolutionStats {
    uint32_t generation;
    float best_fitness;
    float worst_fitness;
    float average_fitness;
    float fitness_variance;
    uint32_t unique_genomes;
    float convergence_rate;
    std::chrono::milliseconds generation_time;
    
    EvolutionStats() : generation(0), best_fitness(0.0f), worst_fitness(0.0f),
                      average_fitness(0.0f), fitness_variance(0.0f),
                      unique_genomes(0), convergence_rate(0.0f),
                      generation_time(std::chrono::milliseconds(0)) {}
};

// Callbacks for evolution monitoring
using FitnessCallback = std::function<void(const EvolutionStats&)>;
using GenerationCallback = std::function<bool(uint32_t generation, const GenomePopulation&)>;
using ConvergenceCallback = std::function<bool(const EvolutionStats&)>;

// Main genetic algorithm class
class GeneticAlgorithm {
public:
    GeneticAlgorithm();
    ~GeneticAlgorithm();
    
    // Initialization
    bool initialize(const EvolutionaryParams& params,
                   const GridDimensions& grid_dims,
                   uint32_t num_inputs,
                   uint32_t num_outputs);
    
    void cleanup();
    
    // Population management
    bool initialize_population(std::mt19937& rng);
    void set_initial_population(const std::vector<Genome>& initial_population);
    
    // Evolution process
    bool evolve(const std::vector<TestCase>& test_cases,
               const FitnessComponents& fitness_weights,
               std::mt19937& rng);
    
    bool evolve_single_generation(const std::vector<TestCase>& test_cases,
                                 const FitnessComponents& fitness_weights,
                                 std::mt19937& rng);
    
    // Results
    const Genome& get_best_genome() const;
    const GenomePopulation& get_population() const;
    const std::vector<EvolutionStats>& get_evolution_history() const;
    
    // Callbacks
    void set_fitness_callback(FitnessCallback callback) { fitness_callback_ = callback; }
    void set_generation_callback(GenerationCallback callback) { generation_callback_ = callback; }
    void set_convergence_callback(ConvergenceCallback callback) { convergence_callback_ = callback; }
    
    // Configuration
    void set_parameters(const EvolutionaryParams& params) { params_ = params; }
    const EvolutionaryParams& get_parameters() const { return params_; }
    
    // GPU acceleration
    bool enable_gpu_acceleration();
    void disable_gpu_acceleration();
    bool is_gpu_enabled() const { return gpu_simulator_ != nullptr; }
    
    // Advanced features
    bool enable_adaptive_parameters();
    void disable_adaptive_parameters();
    bool is_adaptive_enabled() const { return adaptive_parameters_; }
    
    // Diversity maintenance
    bool enable_diversity_maintenance();
    void disable_diversity_maintenance();
    float calculate_population_diversity() const;
    
    // Constraint handling
    void add_constraint(std::function<bool(const Genome&)> constraint);
    void clear_constraints();
    
    // Multi-objective optimization
    bool enable_multi_objective();
    void set_pareto_ranking(bool enabled);
    std::vector<Genome> get_pareto_front() const;
    
private:
    // Core evolution components
    std::unique_ptr<GenomePopulation> population_;
    std::unique_ptr<GenomePopulation> next_generation_;
    
    // Evolution parameters
    EvolutionaryParams params_;
    GridDimensions grid_dims_;
    uint32_t num_inputs_;
    uint32_t num_outputs_;
    
    // GPU acceleration
    std::unique_ptr<GPUSimulator> gpu_simulator_;
    
    // Evolution tracking
    std::vector<EvolutionStats> evolution_history_;
    uint32_t current_generation_;
    
    // Callbacks
    FitnessCallback fitness_callback_;
    GenerationCallback generation_callback_;
    ConvergenceCallback convergence_callback_;
    
    // Advanced features
    bool adaptive_parameters_;
    bool diversity_maintenance_;
    bool multi_objective_;
    bool pareto_ranking_;
    
    // Constraints
    std::vector<std::function<bool(const Genome&)>> constraints_;
    
    // Selection methods
    std::vector<uint32_t> tournament_selection(const GenomePopulation& population,
                                              uint32_t num_parents,
                                              std::mt19937& rng);
    
    std::vector<uint32_t> roulette_wheel_selection(const GenomePopulation& population,
                                                  uint32_t num_parents,
                                                  std::mt19937& rng);
    
    std::vector<uint32_t> rank_based_selection(const GenomePopulation& population,
                                              uint32_t num_parents,
                                              std::mt19937& rng);
    
    std::vector<uint32_t> elitist_selection(const GenomePopulation& population,
                                           uint32_t num_parents,
                                           std::mt19937& rng);
    
    // Crossover methods
    std::pair<Genome, Genome> perform_crossover(const Genome& parent1,
                                               const Genome& parent2,
                                               std::mt19937& rng);
    
    // Mutation methods
    void perform_mutation(Genome& genome, std::mt19937& rng);
    
    // Fitness evaluation
    bool evaluate_fitness(GenomePopulation& population,
                         const std::vector<TestCase>& test_cases,
                         const FitnessComponents& fitness_weights);
    
    // Population management
    void apply_elitism(const GenomePopulation& current_pop,
                      GenomePopulation& next_pop);
    
    void maintain_diversity(GenomePopulation& population,
                           std::mt19937& rng);
    
    // Adaptive parameters
    void update_adaptive_parameters(const EvolutionStats& stats);
    
    // Statistics calculation
    EvolutionStats calculate_generation_stats(const GenomePopulation& population);
    
    // Constraint handling
    bool satisfies_constraints(const Genome& genome) const;
    void repair_genome(Genome& genome, std::mt19937& rng);
    
    // Multi-objective utilities
    bool dominates(const Genome& a, const Genome& b) const;
    std::vector<std::vector<uint32_t>> calculate_pareto_fronts(const GenomePopulation& population) const;
    void assign_crowding_distance(GenomePopulation& population);
    
    // Convergence detection
    bool has_converged(const EvolutionStats& stats) const;
    
    // Utility methods
    float calculate_fitness_variance(const GenomePopulation& population) const;
    uint32_t count_unique_genomes(const GenomePopulation& population) const;
    
    // GPU-CPU synchronization
    bool synchronize_population_with_gpu();
    bool synchronize_population_from_gpu();
};

// Utility classes for specialized evolution strategies

// Island model for parallel evolution
class IslandModel {
public:
    IslandModel(uint32_t num_islands, uint32_t migration_interval);
    
    bool initialize(const EvolutionaryParams& params,
                   const GridDimensions& grid_dims,
                   uint32_t num_inputs,
                   uint32_t num_outputs);
    
    bool evolve(const std::vector<TestCase>& test_cases,
               const FitnessComponents& fitness_weights,
               std::mt19937& rng);
    
    const Genome& get_best_genome() const;
    
private:
    std::vector<std::unique_ptr<GeneticAlgorithm>> islands_;
    uint32_t migration_interval_;
    
    void migrate_individuals(std::mt19937& rng);
};

// Coevolutionary algorithm
class CoevolutionaryAlgorithm {
public:
    CoevolutionaryAlgorithm();
    
    bool initialize(const EvolutionaryParams& params,
                   const GridDimensions& grid_dims,
                   uint32_t num_inputs,
                   uint32_t num_outputs);
    
    bool evolve(const std::vector<TestCase>& test_cases,
               const FitnessComponents& fitness_weights,
               std::mt19937& rng);
    
private:
    std::unique_ptr<GeneticAlgorithm> circuit_population_;
    std::unique_ptr<GeneticAlgorithm> test_population_;
    
    void co_evaluate_populations();
};

// Utility functions
std::unique_ptr<GeneticAlgorithm> create_genetic_algorithm(
    const EvolutionaryParams& params,
    const GridDimensions& grid_dims,
    uint32_t num_inputs,
    uint32_t num_outputs);

EvolutionaryParams get_default_params_for_problem(const std::string& problem_type);

std::vector<TestCase> generate_test_cases_for_problem(const std::string& problem_type,
                                                     const std::vector<uint32_t>& parameters);

} // namespace circuit 