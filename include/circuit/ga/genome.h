#pragma once

#include "../core/types.h"
#include "../core/circuit.h"
#include <vector>
#include <random>
#include <memory>

namespace circuit {

// Genome class for genetic algorithm
class Genome {
public:
    // Constructors
    Genome();
    Genome(const GridDimensions& grid_dims, uint32_t num_inputs, uint32_t num_outputs);
    Genome(const Genome& other);
    Genome(Genome&& other) noexcept;
    
    // Assignment operators
    Genome& operator=(const Genome& other);
    Genome& operator=(Genome&& other) noexcept;
    
    // Destructor
    ~Genome();
    
    // Initialization
    void initialize(const GridDimensions& grid_dims, uint32_t num_inputs, uint32_t num_outputs);
    void randomize(std::mt19937& rng);
    void clear();
    
    // Genome access
    const std::vector<Gene>& get_genes() const { return genes_; }
    std::vector<Gene>& get_genes() { return genes_; }
    const Gene& get_gene(uint32_t index) const { return genes_[index]; }
    Gene& get_gene(uint32_t index) { return genes_[index]; }
    
    // Genome properties
    uint32_t get_gene_count() const { return genes_.size(); }
    uint32_t get_num_inputs() const { return num_inputs_; }
    uint32_t get_num_outputs() const { return num_outputs_; }
    const GridDimensions& get_grid_dimensions() const { return grid_dims_; }
    
    // Fitness and evaluation
    float get_fitness() const { return fitness_; }
    void set_fitness(float fitness) { fitness_ = fitness; }
    const PerformanceMetrics& get_performance_metrics() const { return performance_metrics_; }
    void set_performance_metrics(const PerformanceMetrics& metrics) { performance_metrics_ = metrics; }
    
    // Genome operations
    void mutate(std::mt19937& rng, float mutation_rate);
    static Genome crossover(const Genome& parent1, const Genome& parent2, std::mt19937& rng);
    static std::pair<Genome, Genome> two_point_crossover(const Genome& parent1, const Genome& parent2, std::mt19937& rng);
    static std::pair<Genome, Genome> uniform_crossover(const Genome& parent1, const Genome& parent2, std::mt19937& rng);
    
    // Circuit conversion
    std::unique_ptr<Circuit> to_circuit() const;
    bool from_circuit(const Circuit& circuit);
    
    // Validation
    bool is_valid() const;
    void repair(std::mt19937& rng);
    
    // Serialization
    bool save_to_file(const std::string& filename) const;
    bool load_from_file(const std::string& filename);
    std::string to_string() const;
    
    // Comparison operators
    bool operator==(const Genome& other) const;
    bool operator!=(const Genome& other) const;
    bool operator<(const Genome& other) const;  // For sorting by fitness
    
    // Statistics
    struct GenomeStats {
        uint32_t active_genes;
        uint32_t total_connections;
        uint32_t gate_type_counts[static_cast<int>(GateType::COUNT)];
        float complexity_score;
        
        GenomeStats() : active_genes(0), total_connections(0), complexity_score(0.0f) {
            for (int i = 0; i < static_cast<int>(GateType::COUNT); ++i) {
                gate_type_counts[i] = 0;
            }
        }
    };
    
    GenomeStats get_statistics() const;
    
private:
    // Genome data
    std::vector<Gene> genes_;
    GridDimensions grid_dims_;
    uint32_t num_inputs_;
    uint32_t num_outputs_;
    float fitness_;
    PerformanceMetrics performance_metrics_;
    
    // Mutation operations
    void mutate_gate_type(Gene& gene, std::mt19937& rng);
    void mutate_connections(Gene& gene, std::mt19937& rng);
    void mutate_activation(Gene& gene, std::mt19937& rng);
    void mutate_position(Gene& gene, std::mt19937& rng);
    
    // Helper methods
    uint32_t position_to_index(uint32_t x, uint32_t y) const;
    std::pair<uint32_t, uint32_t> index_to_position(uint32_t index) const;
    bool is_valid_connection(uint32_t from_pos, uint32_t to_pos) const;
    uint32_t get_random_valid_connection(std::mt19937& rng, uint32_t gene_position) const;
    
    // Validation helpers
    bool validate_gene(const Gene& gene) const;
    bool validate_connections() const;
    bool has_feedback_loops() const;
    
    // Repair operations
    void repair_invalid_connections(std::mt19937& rng);
    void repair_duplicate_positions(std::mt19937& rng);
    void repair_missing_io_gates(std::mt19937& rng);
};

// Mutation operators
struct MutationConfig {
    float gate_type_mutation_rate;
    float connection_mutation_rate;
    float activation_mutation_rate;
    float position_mutation_rate;
    float add_gate_rate;
    float remove_gate_rate;
    
    MutationConfig() : gate_type_mutation_rate(0.1f), connection_mutation_rate(0.2f),
                      activation_mutation_rate(0.05f), position_mutation_rate(0.01f),
                      add_gate_rate(0.02f), remove_gate_rate(0.01f) {}
};

// Crossover operators
enum class CrossoverType {
    SINGLE_POINT,
    TWO_POINT,
    UNIFORM,
    GRID_BASED
};

struct CrossoverConfig {
    CrossoverType type;
    float crossover_rate;
    
    CrossoverConfig() : type(CrossoverType::TWO_POINT), crossover_rate(0.8f) {}
};

// Genome population utilities
class GenomePopulation {
public:
    GenomePopulation(uint32_t population_size, const GridDimensions& grid_dims,
                    uint32_t num_inputs, uint32_t num_outputs);
    
    // Population management
    void initialize_random(std::mt19937& rng);
    void clear();
    void resize(uint32_t new_size);
    
    // Access
    Genome& operator[](uint32_t index) { return population_[index]; }
    const Genome& operator[](uint32_t index) const { return population_[index]; }
    uint32_t size() const { return population_.size(); }
    
    // Iterators
    std::vector<Genome>::iterator begin() { return population_.begin(); }
    std::vector<Genome>::iterator end() { return population_.end(); }
    std::vector<Genome>::const_iterator begin() const { return population_.begin(); }
    std::vector<Genome>::const_iterator end() const { return population_.end(); }
    
    // Statistics
    float get_best_fitness() const;
    float get_worst_fitness() const;
    float get_average_fitness() const;
    const Genome& get_best_genome() const;
    const Genome& get_worst_genome() const;
    
    // Sorting
    void sort_by_fitness();
    
private:
    std::vector<Genome> population_;
    GridDimensions grid_dims_;
    uint32_t num_inputs_;
    uint32_t num_outputs_;
};

// Utility functions
std::unique_ptr<Genome> create_random_genome(const GridDimensions& grid_dims,
                                           uint32_t num_inputs, uint32_t num_outputs,
                                           std::mt19937& rng);

std::unique_ptr<Genome> create_minimal_genome(const GridDimensions& grid_dims,
                                            uint32_t num_inputs, uint32_t num_outputs);

float calculate_genome_similarity(const Genome& genome1, const Genome& genome2);

} // namespace circuit 