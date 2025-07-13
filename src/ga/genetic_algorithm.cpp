#include "circuit/ga/genetic_algorithm.h"
#include "circuit/core/types.h"
#include "circuit/utils/utils.h"
#include "circuit/gpu/gpu_simulator.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cassert>
#include <iostream>

namespace circuit {

// GeneticAlgorithm implementation
GeneticAlgorithm::GeneticAlgorithm() 
    : current_generation_(0), adaptive_parameters_(false), diversity_maintenance_(false),
      multi_objective_(false), pareto_ranking_(false) {}

GeneticAlgorithm::~GeneticAlgorithm() {
    cleanup();
}

bool GeneticAlgorithm::initialize(const EvolutionaryParams& params,
                                 const GridDimensions& grid_dims,
                                 uint32_t num_inputs,
                                 uint32_t num_outputs) {
    params_ = params;
    grid_dims_ = grid_dims;
    num_inputs_ = num_inputs;
    num_outputs_ = num_outputs;
    current_generation_ = 0;
    
    // Initialize population
    population_ = std::make_unique<GenomePopulation>(params_.population_size, grid_dims, num_inputs, num_outputs);
    next_generation_ = std::make_unique<GenomePopulation>(params_.population_size, grid_dims, num_inputs, num_outputs);
    
    // Clear history
    evolution_history_.clear();
    
    return true;
}

void GeneticAlgorithm::cleanup() {
    population_.reset();
    next_generation_.reset();
    gpu_simulator_.reset();
    evolution_history_.clear();
}

bool GeneticAlgorithm::initialize_population(std::mt19937& rng) {
    if (!population_) {
        return false;
    }
    
    population_->initialize_random(rng);
    return true;
}

void GeneticAlgorithm::set_initial_population(const std::vector<Genome>& initial_population) {
    if (!population_ || initial_population.size() != population_->size()) {
        return;
    }
    
    for (uint32_t i = 0; i < initial_population.size(); ++i) {
        (*population_)[i] = initial_population[i];
    }
}

bool GeneticAlgorithm::evolve(const std::vector<TestCase>& test_cases,
                             const FitnessComponents& fitness_weights,
                             std::mt19937& rng) {
    if (!population_) {
        return false;
    }
    
    // Initialize population if needed
    if (current_generation_ == 0) {
        initialize_population(rng);
    }
    
    // Evolution loop
    for (uint32_t generation = 0; generation < params_.max_generations; ++generation) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Evaluate fitness
        if (!evaluate_fitness(*population_, test_cases, fitness_weights)) {
            return false;
        }
        
        // Calculate statistics
        auto stats = calculate_generation_stats(*population_);
        stats.generation = generation;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        stats.generation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        evolution_history_.push_back(stats);
        
        // Call callbacks
        if (fitness_callback_) {
            fitness_callback_(stats);
        }
        
        if (generation_callback_) {
            if (!generation_callback_(generation, *population_)) {
                break; // Early termination requested
            }
        }
        
        // Check convergence
        if (has_converged(stats)) {
            break;
        }
        
        // Perform single generation evolution
        if (!evolve_single_generation(test_cases, fitness_weights, rng)) {
            return false;
        }
        
        current_generation_++;
    }
    
    return true;
}

bool GeneticAlgorithm::evolve_single_generation(const std::vector<TestCase>& test_cases,
                                               const FitnessComponents& fitness_weights,
                                               std::mt19937& rng) {
    if (!population_ || !next_generation_) {
        return false;
    }
    
    // Sort population by fitness
    population_->sort_by_fitness();
    
    // Apply elitism
    apply_elitism(*population_, *next_generation_);
    
    // Generate offspring
    uint32_t offspring_count = params_.population_size - params_.elite_count;
    uint32_t offspring_generated = 0;
    
    while (offspring_generated < offspring_count) {
        // Select parents
        std::vector<uint32_t> parent_indices;
        
        switch (params_.selection_strategy) {
            case SelectionStrategy::TOURNAMENT:
                parent_indices = tournament_selection(*population_, 2, rng);
                break;
            case SelectionStrategy::ROULETTE_WHEEL:
                parent_indices = roulette_wheel_selection(*population_, 2, rng);
                break;
            case SelectionStrategy::RANK_BASED:
                parent_indices = rank_based_selection(*population_, 2, rng);
                break;
            case SelectionStrategy::ELITIST:
                parent_indices = elitist_selection(*population_, 2, rng);
                break;
        }
        
        if (parent_indices.size() < 2) {
            return false;
        }
        
        const Genome& parent1 = (*population_)[parent_indices[0]];
        const Genome& parent2 = (*population_)[parent_indices[1]];
        
        // Perform crossover
        std::uniform_real_distribution<float> crossover_prob(0.0f, 1.0f);
        if (crossover_prob(rng) < params_.crossover_rate) {
            auto offspring_pair = perform_crossover(parent1, parent2, rng);
            
            // Mutate offspring
            perform_mutation(offspring_pair.first, rng);
            perform_mutation(offspring_pair.second, rng);
            
            // Add to next generation
            if (offspring_generated < offspring_count) {
                (*next_generation_)[params_.elite_count + offspring_generated] = std::move(offspring_pair.first);
                offspring_generated++;
            }
            
            if (offspring_generated < offspring_count) {
                (*next_generation_)[params_.elite_count + offspring_generated] = std::move(offspring_pair.second);
                offspring_generated++;
            }
        } else {
            // Just copy parents with mutation
            Genome offspring1 = parent1;
            Genome offspring2 = parent2;
            
            perform_mutation(offspring1, rng);
            perform_mutation(offspring2, rng);
            
            if (offspring_generated < offspring_count) {
                (*next_generation_)[params_.elite_count + offspring_generated] = std::move(offspring1);
                offspring_generated++;
            }
            
            if (offspring_generated < offspring_count) {
                (*next_generation_)[params_.elite_count + offspring_generated] = std::move(offspring2);
                offspring_generated++;
            }
        }
    }
    
    // Swap populations
    std::swap(population_, next_generation_);
    
    // Maintain diversity if enabled
    if (diversity_maintenance_) {
        maintain_diversity(*population_, rng);
    }
    
    // Update adaptive parameters if enabled
    if (adaptive_parameters_) {
        auto stats = calculate_generation_stats(*population_);
        update_adaptive_parameters(stats);
    }
    
    return true;
}

const Genome& GeneticAlgorithm::get_best_genome() const {
    if (!population_) {
        static Genome dummy;
        return dummy;
    }
    
    return population_->get_best_genome();
}

const GenomePopulation& GeneticAlgorithm::get_population() const {
    static GenomePopulation dummy(1, GridDimensions{1, 1}, 1, 1);
    return population_ ? *population_ : dummy;
}

const std::vector<EvolutionStats>& GeneticAlgorithm::get_evolution_history() const {
    return evolution_history_;
}

// Selection methods
std::vector<uint32_t> GeneticAlgorithm::tournament_selection(const GenomePopulation& population,
                                                           uint32_t num_parents,
                                                           std::mt19937& rng) {
    std::vector<uint32_t> selected;
    selected.reserve(num_parents);
    
    std::uniform_int_distribution<uint32_t> pop_dist(0, population.size() - 1);
    
    for (uint32_t p = 0; p < num_parents; ++p) {
        // Tournament selection
        uint32_t best_idx = pop_dist(rng);
        float best_fitness = population[best_idx].get_fitness();
        
        for (uint32_t i = 1; i < params_.tournament_size; ++i) {
            uint32_t candidate_idx = pop_dist(rng);
            float candidate_fitness = population[candidate_idx].get_fitness();
            
            if (candidate_fitness > best_fitness) {
                best_idx = candidate_idx;
                best_fitness = candidate_fitness;
            }
        }
        
        selected.push_back(best_idx);
    }
    
    return selected;
}

std::vector<uint32_t> GeneticAlgorithm::roulette_wheel_selection(const GenomePopulation& population,
                                                               uint32_t num_parents,
                                                               std::mt19937& rng) {
    std::vector<uint32_t> selected;
    selected.reserve(num_parents);
    
    // Calculate fitness sum
    float total_fitness = 0.0f;
    for (uint32_t i = 0; i < population.size(); ++i) {
        total_fitness += std::max(0.0f, population[i].get_fitness());
    }
    
    if (total_fitness <= 0.0f) {
        // Fallback to random selection
        std::uniform_int_distribution<uint32_t> pop_dist(0, population.size() - 1);
        for (uint32_t p = 0; p < num_parents; ++p) {
            selected.push_back(pop_dist(rng));
        }
        return selected;
    }
    
    std::uniform_real_distribution<float> wheel_dist(0.0f, total_fitness);
    
    for (uint32_t p = 0; p < num_parents; ++p) {
        float wheel_pos = wheel_dist(rng);
        float current_sum = 0.0f;
        
        for (uint32_t i = 0; i < population.size(); ++i) {
            current_sum += std::max(0.0f, population[i].get_fitness());
            if (current_sum >= wheel_pos) {
                selected.push_back(i);
                break;
            }
        }
    }
    
    return selected;
}

std::vector<uint32_t> GeneticAlgorithm::rank_based_selection(const GenomePopulation& population,
                                                           uint32_t num_parents,
                                                           std::mt19937& rng) {
    std::vector<uint32_t> selected;
    selected.reserve(num_parents);
    
    // Create ranking
    std::vector<std::pair<float, uint32_t>> fitness_rank;
    for (uint32_t i = 0; i < population.size(); ++i) {
        fitness_rank.emplace_back(population[i].get_fitness(), i);
    }
    
    std::sort(fitness_rank.begin(), fitness_rank.end(), std::greater<std::pair<float, uint32_t>>());
    
    // Rank-based selection probabilities
    float total_rank = population.size() * (population.size() + 1) / 2.0f;
    std::uniform_real_distribution<float> rank_dist(0.0f, total_rank);
    
    for (uint32_t p = 0; p < num_parents; ++p) {
        float rank_pos = rank_dist(rng);
        float current_sum = 0.0f;
        
        for (uint32_t i = 0; i < population.size(); ++i) {
            current_sum += (population.size() - i);
            if (current_sum >= rank_pos) {
                selected.push_back(fitness_rank[i].second);
                break;
            }
        }
    }
    
    return selected;
}

std::vector<uint32_t> GeneticAlgorithm::elitist_selection(const GenomePopulation& population,
                                                        uint32_t num_parents,
                                                        std::mt19937& rng) {
    std::vector<uint32_t> selected;
    selected.reserve(num_parents);
    
    // Create sorted indices by fitness
    std::vector<std::pair<float, uint32_t>> fitness_indices;
    for (uint32_t i = 0; i < population.size(); ++i) {
        fitness_indices.emplace_back(population[i].get_fitness(), i);
    }
    
    std::sort(fitness_indices.begin(), fitness_indices.end(), std::greater<std::pair<float, uint32_t>>());
    
    // Select from top performers
    uint32_t elite_pool_size = std::min(params_.elite_count * 2, static_cast<uint32_t>(population.size()));
    std::uniform_int_distribution<uint32_t> elite_dist(0, elite_pool_size - 1);
    
    for (uint32_t p = 0; p < num_parents; ++p) {
        uint32_t elite_idx = elite_dist(rng);
        selected.push_back(fitness_indices[elite_idx].second);
    }
    
    return selected;
}

// Crossover methods
std::pair<Genome, Genome> GeneticAlgorithm::perform_crossover(const Genome& parent1,
                                                             const Genome& parent2,
                                                             std::mt19937& rng) {
    switch (params_.crossover_type) {
        case CrossoverType::SINGLE_POINT:
            return Genome::two_point_crossover(parent1, parent2, rng); // Use two-point as fallback
        case CrossoverType::TWO_POINT:
            return Genome::two_point_crossover(parent1, parent2, rng);
        case CrossoverType::UNIFORM:
            return Genome::uniform_crossover(parent1, parent2, rng);
        case CrossoverType::GRID_BASED:
            return Genome::uniform_crossover(parent1, parent2, rng); // Use uniform as fallback
        default:
            return Genome::two_point_crossover(parent1, parent2, rng);
    }
}

// Mutation methods
void GeneticAlgorithm::perform_mutation(Genome& genome, std::mt19937& rng) {
    genome.mutate(rng, params_.mutation_rate);
}

// Fitness evaluation
bool GeneticAlgorithm::evaluate_fitness(GenomePopulation& population,
                                       const std::vector<TestCase>& test_cases,
                                       const FitnessComponents& fitness_weights) {
    // For now, implement CPU-based fitness evaluation
    // GPU acceleration would be added later
    
    for (uint32_t i = 0; i < population.size(); ++i) {
        Genome& genome = population[i];
        
        // Convert genome to circuit
        auto circuit = genome.to_circuit();
        if (!circuit) {
            genome.set_fitness(0.0f);
            continue;
        }
        
        // Evaluate circuit on test cases
        float correctness = 0.0f;
        float total_delay = 0.0f;
        float total_power = 0.0f;
        float total_area = 0.0f;
        
        for (const auto& test_case : test_cases) {
            // Convert LogicState to SignalValue
            std::vector<SignalValue> inputs;
            for (const auto& state : test_case.inputs) {
                inputs.push_back(static_cast<SignalValue>(state));
            }
            
            // Simulate circuit with test case
            std::vector<SignalValue> outputs;
            bool success = circuit->simulate(inputs, outputs);
            
            // Check correctness
            bool correct = true;
            if (success && outputs.size() == test_case.expected_outputs.size()) {
                for (uint32_t j = 0; j < outputs.size(); ++j) {
                    if (outputs[j] != static_cast<SignalValue>(test_case.expected_outputs[j])) {
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
            
            // Accumulate performance metrics
            if (success) {
                auto metrics = circuit->evaluate_performance({test_case});
                total_delay += metrics.total_delay;
                total_power += metrics.power_consumption;
                total_area += metrics.area_cost;
            }
        }
        
        // Normalize metrics
        correctness /= test_cases.size();
        total_delay /= test_cases.size();
        total_power /= test_cases.size();
        total_area /= test_cases.size();
        
        // Calculate composite fitness
        float fitness = fitness_weights.correctness_weight * correctness;
        
        // Add penalty for delay (lower is better)
        if (total_delay > 0.0f) {
            fitness -= fitness_weights.delay_weight * total_delay;
        }
        
        // Add penalty for power (lower is better)
        if (total_power > 0.0f) {
            fitness -= fitness_weights.power_weight * total_power;
        }
        
        // Add penalty for area (lower is better)
        if (total_area > 0.0f) {
            fitness -= fitness_weights.area_weight * total_area;
        }
        
        // Ensure fitness is non-negative
        fitness = std::max(0.0f, fitness);
        
        genome.set_fitness(fitness);
        
        // Set performance metrics
        PerformanceMetrics metrics;
        metrics.correctness = correctness;
        metrics.propagation_delay = total_delay;
        metrics.power_consumption = total_power;
        metrics.area_cost = total_area;
        genome.set_performance_metrics(metrics);
    }
    
    return true;
}

// Population management
void GeneticAlgorithm::apply_elitism(const GenomePopulation& current_pop,
                                    GenomePopulation& next_pop) {
    // Copy best individuals to next generation
    std::vector<std::pair<float, uint32_t>> fitness_indices;
    for (uint32_t i = 0; i < current_pop.size(); ++i) {
        fitness_indices.emplace_back(current_pop[i].get_fitness(), i);
    }
    
    std::sort(fitness_indices.begin(), fitness_indices.end(), std::greater<std::pair<float, uint32_t>>());
    
    for (uint32_t i = 0; i < params_.elite_count && i < current_pop.size(); ++i) {
        next_pop[i] = current_pop[fitness_indices[i].second];
    }
}

void GeneticAlgorithm::maintain_diversity(GenomePopulation& population, std::mt19937& rng) {
    // Simple diversity maintenance by replacing similar individuals
    const float similarity_threshold = 0.9f;
    
    for (uint32_t i = 0; i < population.size(); ++i) {
        for (uint32_t j = i + 1; j < population.size(); ++j) {
            float similarity = calculate_genome_similarity(population[i], population[j]);
            
            if (similarity > similarity_threshold) {
                // Replace the worse-performing individual
                if (population[i].get_fitness() < population[j].get_fitness()) {
                    population[i] = Genome(grid_dims_, num_inputs_, num_outputs_);
                    population[i].randomize(rng);
                } else {
                    population[j] = Genome(grid_dims_, num_inputs_, num_outputs_);
                    population[j].randomize(rng);
                }
            }
        }
    }
}

// Statistics calculation
EvolutionStats GeneticAlgorithm::calculate_generation_stats(const GenomePopulation& population) {
    EvolutionStats stats;
    
    if (population.size() == 0) {
        return stats;
    }
    
    stats.best_fitness = population.get_best_fitness();
    stats.worst_fitness = population.get_worst_fitness();
    stats.average_fitness = population.get_average_fitness();
    stats.fitness_variance = calculate_fitness_variance(population);
    stats.unique_genomes = count_unique_genomes(population);
    stats.convergence_rate = (stats.best_fitness - stats.average_fitness) / (stats.best_fitness + 1e-6f);
    
    return stats;
}

// Adaptive parameters
void GeneticAlgorithm::update_adaptive_parameters(const EvolutionStats& stats) {
    // Adjust mutation rate based on convergence
    if (stats.convergence_rate < 0.1f) {
        // Population is converging, increase mutation rate
        params_.mutation_rate = std::min(0.5f, params_.mutation_rate * 1.1f);
    } else if (stats.convergence_rate > 0.5f) {
        // Population is diverse, decrease mutation rate
        params_.mutation_rate = std::max(0.01f, params_.mutation_rate * 0.9f);
    }
    
    // Adjust crossover rate based on diversity
    if (stats.unique_genomes < population_->size() * 0.3f) {
        // Low diversity, increase crossover rate
        params_.crossover_rate = std::min(0.95f, params_.crossover_rate * 1.05f);
    }
}

// Convergence detection
bool GeneticAlgorithm::has_converged(const EvolutionStats& stats) const {
    // Check if fitness has plateaued
    if (evolution_history_.size() >= 10) {
        float recent_improvement = stats.best_fitness - evolution_history_[evolution_history_.size() - 10].best_fitness;
        if (recent_improvement < 0.001f) {
            return true;
        }
    }
    
    // Check if population has converged
    if (stats.unique_genomes < population_->size() * 0.1f) {
        return true;
    }
    
    return false;
}

// Utility methods
float GeneticAlgorithm::calculate_fitness_variance(const GenomePopulation& population) const {
    float mean = population.get_average_fitness();
    float variance = 0.0f;
    
    for (uint32_t i = 0; i < population.size(); ++i) {
        float diff = population[i].get_fitness() - mean;
        variance += diff * diff;
    }
    
    return variance / population.size();
}

uint32_t GeneticAlgorithm::count_unique_genomes(const GenomePopulation& population) const {
    std::vector<bool> unique(population.size(), true);
    uint32_t count = 0;
    
    for (uint32_t i = 0; i < population.size(); ++i) {
        if (!unique[i]) continue;
        
        count++;
        for (uint32_t j = i + 1; j < population.size(); ++j) {
            if (unique[j] && population[i] == population[j]) {
                unique[j] = false;
            }
        }
    }
    
    return count;
}

// GPU acceleration methods (stubs for now)
bool GeneticAlgorithm::enable_gpu_acceleration() {
    // GPU acceleration will be implemented in Stage 2
    return false;
}

void GeneticAlgorithm::disable_gpu_acceleration() {
    gpu_simulator_.reset();
}

// Advanced features (stubs for now)
bool GeneticAlgorithm::enable_adaptive_parameters() {
    adaptive_parameters_ = true;
    return true;
}

void GeneticAlgorithm::disable_adaptive_parameters() {
    adaptive_parameters_ = false;
}

bool GeneticAlgorithm::enable_diversity_maintenance() {
    diversity_maintenance_ = true;
    return true;
}

void GeneticAlgorithm::disable_diversity_maintenance() {
    diversity_maintenance_ = false;
}

float GeneticAlgorithm::calculate_population_diversity() const {
    if (!population_ || population_->size() < 2) {
        return 0.0f;
    }
    
    float total_diversity = 0.0f;
    uint32_t comparisons = 0;
    
    for (uint32_t i = 0; i < population_->size(); ++i) {
        for (uint32_t j = i + 1; j < population_->size(); ++j) {
            float similarity = calculate_genome_similarity((*population_)[i], (*population_)[j]);
            total_diversity += (1.0f - similarity);
            comparisons++;
        }
    }
    
    return comparisons > 0 ? total_diversity / comparisons : 0.0f;
}

// Constraint handling (stubs for now)
void GeneticAlgorithm::add_constraint(std::function<bool(const Genome&)> constraint) {
    constraints_.push_back(constraint);
}

void GeneticAlgorithm::clear_constraints() {
    constraints_.clear();
}

bool GeneticAlgorithm::satisfies_constraints(const Genome& genome) const {
    for (const auto& constraint : constraints_) {
        if (!constraint(genome)) {
            return false;
        }
    }
    return true;
}

void GeneticAlgorithm::repair_genome(Genome& genome, std::mt19937& rng) {
    genome.repair(rng);
}

// Multi-objective optimization (stubs for now)
bool GeneticAlgorithm::enable_multi_objective() {
    multi_objective_ = true;
    return true;
}

void GeneticAlgorithm::set_pareto_ranking(bool enabled) {
    pareto_ranking_ = enabled;
}

std::vector<Genome> GeneticAlgorithm::get_pareto_front() const {
    // Simple implementation - return top performers
    std::vector<Genome> pareto_front;
    if (population_) {
        pareto_front.push_back(population_->get_best_genome());
    }
    return pareto_front;
}

// Utility functions
std::unique_ptr<GeneticAlgorithm> create_genetic_algorithm(const EvolutionaryParams& params,
                                                         const GridDimensions& grid_dims,
                                                         uint32_t num_inputs,
                                                         uint32_t num_outputs) {
    auto ga = std::make_unique<GeneticAlgorithm>();
    if (ga->initialize(params, grid_dims, num_inputs, num_outputs)) {
        return ga;
    }
    return nullptr;
}

EvolutionaryParams get_default_params_for_problem(const std::string& problem_type) {
    EvolutionaryParams params;
    
    if (problem_type == "adder") {
        params.population_size = 100;
        params.max_generations = 500;
        params.mutation_rate = 0.1f;
        params.crossover_rate = 0.8f;
        params.tournament_size = 4;
        params.elite_count = 5;
    } else if (problem_type == "multiplexer") {
        params.population_size = 150;
        params.max_generations = 800;
        params.mutation_rate = 0.15f;
        params.crossover_rate = 0.75f;
        params.tournament_size = 6;
        params.elite_count = 8;
    } else if (problem_type == "comparator") {
        params.population_size = 80;
        params.max_generations = 400;
        params.mutation_rate = 0.08f;
        params.crossover_rate = 0.85f;
        params.tournament_size = 3;
        params.elite_count = 4;
    }
    
    return params;
}

} // namespace circuit 