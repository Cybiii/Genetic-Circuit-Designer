#include "circuit/ga/genome.h"
#include "circuit/core/types.h"
#include <algorithm>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <sstream>
#include <cassert>

namespace circuit {

// Constructors
Genome::Genome() : num_inputs_(0), num_outputs_(0), fitness_(0.0f) {}

Genome::Genome(const GridDimensions& grid_dims, uint32_t num_inputs, uint32_t num_outputs)
    : grid_dims_(grid_dims), num_inputs_(num_inputs), num_outputs_(num_outputs), fitness_(0.0f) {
    initialize(grid_dims, num_inputs, num_outputs);
}

Genome::Genome(const Genome& other) 
    : genes_(other.genes_), grid_dims_(other.grid_dims_), num_inputs_(other.num_inputs_),
      num_outputs_(other.num_outputs_), fitness_(other.fitness_), 
      performance_metrics_(other.performance_metrics_) {}

Genome::Genome(Genome&& other) noexcept 
    : genes_(std::move(other.genes_)), grid_dims_(other.grid_dims_), 
      num_inputs_(other.num_inputs_), num_outputs_(other.num_outputs_),
      fitness_(other.fitness_), performance_metrics_(std::move(other.performance_metrics_)) {}

// Assignment operators
Genome& Genome::operator=(const Genome& other) {
    if (this != &other) {
        genes_ = other.genes_;
        grid_dims_ = other.grid_dims_;
        num_inputs_ = other.num_inputs_;
        num_outputs_ = other.num_outputs_;
        fitness_ = other.fitness_;
        performance_metrics_ = other.performance_metrics_;
    }
    return *this;
}

Genome& Genome::operator=(Genome&& other) noexcept {
    if (this != &other) {
        genes_ = std::move(other.genes_);
        grid_dims_ = other.grid_dims_;
        num_inputs_ = other.num_inputs_;
        num_outputs_ = other.num_outputs_;
        fitness_ = other.fitness_;
        performance_metrics_ = std::move(other.performance_metrics_);
    }
    return *this;
}

// Destructor
Genome::~Genome() = default;

// Initialization
void Genome::initialize(const GridDimensions& grid_dims, uint32_t num_inputs, uint32_t num_outputs) {
    grid_dims_ = grid_dims;
    num_inputs_ = num_inputs;
    num_outputs_ = num_outputs;
    
    // Reserve space for genes (each grid cell can potentially have a gate)
    genes_.reserve(grid_dims.width * grid_dims.height);
    genes_.clear();
    
    // Initialize with empty genes
    for (uint32_t y = 0; y < grid_dims.height; ++y) {
        for (uint32_t x = 0; x < grid_dims.width; ++x) {
            Gene gene;
            gene.position = {x, y};
            gene.gate_type = GateType::NONE;
            gene.is_active = false;
            gene.input_connections.clear();
            gene.output_connections.clear();
            genes_.push_back(gene);
        }
    }
    
    fitness_ = 0.0f;
    performance_metrics_ = PerformanceMetrics();
}

void Genome::randomize(std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> gate_type_dist(0, static_cast<int>(GateType::COUNT) - 1);
    
    // Probability of activating a gate
    const float activation_probability = 0.3f;
    
    for (auto& gene : genes_) {
        // Randomly activate gates
        if (prob_dist(rng) < activation_probability) {
            gene.is_active = true;
            
            // Randomly assign gate type (excluding NONE)
            do {
                gene.gate_type = static_cast<GateType>(gate_type_dist(rng));
            } while (gene.gate_type == GateType::NONE);
            
            // Generate random connections
            uint32_t num_inputs = get_gate_input_count(gene.gate_type);
            gene.input_connections.clear();
            
            for (uint32_t i = 0; i < num_inputs; ++i) {
                uint32_t connection = get_random_valid_connection(rng, position_to_index(gene.position.x, gene.position.y));
                gene.input_connections.push_back(connection);
            }
        } else {
            gene.is_active = false;
            gene.gate_type = GateType::NONE;
            gene.input_connections.clear();
        }
        gene.output_connections.clear(); // Will be populated later
    }
    
    // Update output connections based on input connections
    for (uint32_t i = 0; i < genes_.size(); ++i) {
        for (uint32_t input_conn : genes_[i].input_connections) {
            if (input_conn < genes_.size()) {
                genes_[input_conn].output_connections.push_back(i);
            }
        }
    }
    
    // Ensure we have at least some active gates
    if (std::none_of(genes_.begin(), genes_.end(), [](const Gene& g) { return g.is_active; })) {
        // Activate a few random gates
        std::uniform_int_distribution<uint32_t> pos_dist(0, genes_.size() - 1);
        for (int i = 0; i < 5; ++i) {
            uint32_t idx = pos_dist(rng);
            genes_[idx].is_active = true;
            genes_[idx].gate_type = GateType::AND; // Default to AND gate
        }
    }
    
    repair(rng);
}

void Genome::clear() {
    for (auto& gene : genes_) {
        gene.is_active = false;
        gene.gate_type = GateType::NONE;
        gene.input_connections.clear();
        gene.output_connections.clear();
    }
    fitness_ = 0.0f;
    performance_metrics_ = PerformanceMetrics();
}

// Genome operations
void Genome::mutate(std::mt19937& rng, float mutation_rate) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (auto& gene : genes_) {
        if (prob_dist(rng) < mutation_rate) {
            // Choose mutation type
            std::uniform_int_distribution<int> mutation_type_dist(0, 3);
            int mutation_type = mutation_type_dist(rng);
            
            switch (mutation_type) {
                case 0: mutate_gate_type(gene, rng); break;
                case 1: mutate_connections(gene, rng); break;
                case 2: mutate_activation(gene, rng); break;
                case 3: mutate_position(gene, rng); break;
            }
        }
    }
    
    repair(rng);
}

Genome Genome::crossover(const Genome& parent1, const Genome& parent2, std::mt19937& rng) {
    assert(parent1.genes_.size() == parent2.genes_.size());
    
    Genome offspring(parent1.grid_dims_, parent1.num_inputs_, parent1.num_outputs_);
    
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    // Uniform crossover
    for (uint32_t i = 0; i < offspring.genes_.size(); ++i) {
        if (prob_dist(rng) < 0.5f) {
            offspring.genes_[i] = parent1.genes_[i];
        } else {
            offspring.genes_[i] = parent2.genes_[i];
        }
    }
    
    offspring.repair(rng);
    return offspring;
}

std::pair<Genome, Genome> Genome::two_point_crossover(const Genome& parent1, const Genome& parent2, std::mt19937& rng) {
    assert(parent1.genes_.size() == parent2.genes_.size());
    
    Genome offspring1(parent1.grid_dims_, parent1.num_inputs_, parent1.num_outputs_);
    Genome offspring2(parent1.grid_dims_, parent1.num_inputs_, parent1.num_outputs_);
    
    std::uniform_int_distribution<uint32_t> pos_dist(0, parent1.genes_.size() - 1);
    uint32_t point1 = pos_dist(rng);
    uint32_t point2 = pos_dist(rng);
    
    if (point1 > point2) std::swap(point1, point2);
    
    for (uint32_t i = 0; i < offspring1.genes_.size(); ++i) {
        if (i < point1 || i > point2) {
            offspring1.genes_[i] = parent1.genes_[i];
            offspring2.genes_[i] = parent2.genes_[i];
        } else {
            offspring1.genes_[i] = parent2.genes_[i];
            offspring2.genes_[i] = parent1.genes_[i];
        }
    }
    
    offspring1.repair(rng);
    offspring2.repair(rng);
    
    return std::make_pair(std::move(offspring1), std::move(offspring2));
}

std::pair<Genome, Genome> Genome::uniform_crossover(const Genome& parent1, const Genome& parent2, std::mt19937& rng) {
    assert(parent1.genes_.size() == parent2.genes_.size());
    
    Genome offspring1(parent1.grid_dims_, parent1.num_inputs_, parent1.num_outputs_);
    Genome offspring2(parent1.grid_dims_, parent1.num_inputs_, parent1.num_outputs_);
    
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (uint32_t i = 0; i < offspring1.genes_.size(); ++i) {
        if (prob_dist(rng) < 0.5f) {
            offspring1.genes_[i] = parent1.genes_[i];
            offspring2.genes_[i] = parent2.genes_[i];
        } else {
            offspring1.genes_[i] = parent2.genes_[i];
            offspring2.genes_[i] = parent1.genes_[i];
        }
    }
    
    offspring1.repair(rng);
    offspring2.repair(rng);
    
    return std::make_pair(std::move(offspring1), std::move(offspring2));
}

// Circuit conversion
std::unique_ptr<Circuit> Genome::to_circuit() const {
    auto circuit = std::make_unique<Circuit>(grid_dims_);
    
    // Initialize with specified input and output counts
    // This should be done after construction
    
    // Add gates from active genes
    std::unordered_map<uint32_t, GateId> position_to_gate_id;
    for (uint32_t i = 0; i < genes_.size(); ++i) {
        const auto& gene = genes_[i];
        if (gene.is_active && gene.gate_type != GateType::NONE) {
            GateId gate_id = circuit->add_gate(gene.gate_type, gene.position.x, gene.position.y);
            position_to_gate_id[i] = gate_id;
        }
    }
    
    // Add connections
    for (uint32_t i = 0; i < genes_.size(); ++i) {
        if (genes_[i].is_active) {
            auto to_gate_it = position_to_gate_id.find(i);
            if (to_gate_it != position_to_gate_id.end()) {
                for (uint32_t input_conn : genes_[i].input_connections) {
                    if (input_conn < genes_.size()) {
                        auto from_gate_it = position_to_gate_id.find(input_conn);
                        if (from_gate_it != position_to_gate_id.end()) {
                            circuit->add_connection(from_gate_it->second, 0, to_gate_it->second, 0);
                        }
                    }
                }
            }
        }
    }
    
    return circuit;
}

bool Genome::from_circuit(const Circuit& circuit) {
    // Clear existing genes
    clear();
    
    // This would need to be implemented based on Circuit's interface
    // For now, return true as a stub
    return true;
}

// Validation
bool Genome::is_valid() const {
    // Check basic constraints
    if (genes_.size() != grid_dims_.width * grid_dims_.height) {
        return false;
    }
    
    // Check each gene
    for (const auto& gene : genes_) {
        if (!validate_gene(gene)) {
            return false;
        }
    }
    
    // Check connections
    if (!validate_connections()) {
        return false;
    }
    
    // Check for feedback loops
    if (has_feedback_loops()) {
        return false;
    }
    
    return true;
}

void Genome::repair(std::mt19937& rng) {
    repair_invalid_connections(rng);
    repair_duplicate_positions(rng);
    repair_missing_io_gates(rng);
}

// Comparison operators
bool Genome::operator==(const Genome& other) const {
    return genes_ == other.genes_ && 
           grid_dims_.width == other.grid_dims_.width &&
           grid_dims_.height == other.grid_dims_.height &&
           num_inputs_ == other.num_inputs_ &&
           num_outputs_ == other.num_outputs_;
}

bool Genome::operator!=(const Genome& other) const {
    return !(*this == other);
}

bool Genome::operator<(const Genome& other) const {
    return fitness_ < other.fitness_;
}

// Statistics
Genome::GenomeStats Genome::get_statistics() const {
    GenomeStats stats;
    
    for (const auto& gene : genes_) {
        if (gene.is_active) {
            stats.active_genes++;
            stats.total_connections += gene.input_connections.size();
            
            if (gene.gate_type != GateType::NONE) {
                stats.gate_type_counts[static_cast<int>(gene.gate_type)]++;
            }
        }
    }
    
    // Simple complexity score based on active gates and connections
    stats.complexity_score = stats.active_genes * 1.0f + stats.total_connections * 0.5f;
    
    return stats;
}

// Serialization
bool Genome::save_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Save basic info
    file << grid_dims_.width << " " << grid_dims_.height << "\n";
    file << num_inputs_ << " " << num_outputs_ << "\n";
    file << fitness_ << "\n";
    
    // Save genes
    for (const auto& gene : genes_) {
        file << gene.position.x << " " << gene.position.y << " ";
        file << static_cast<int>(gene.gate_type) << " ";
        file << gene.is_active << " ";
        file << gene.input_connections.size() << " ";
        for (uint32_t conn : gene.input_connections) {
            file << conn << " ";
        }
        file << "\n";
    }
    
    return true;
}

bool Genome::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Load basic info
    file >> grid_dims_.width >> grid_dims_.height;
    file >> num_inputs_ >> num_outputs_;
    file >> fitness_;
    
    // Initialize genes
    initialize(grid_dims_, num_inputs_, num_outputs_);
    
    // Load genes
    for (auto& gene : genes_) {
        int gate_type_int;
        file >> gene.position.x >> gene.position.y;
        file >> gate_type_int;
        file >> gene.is_active;
        
        gene.gate_type = static_cast<GateType>(gate_type_int);
        
        uint32_t num_connections;
        file >> num_connections;
        gene.input_connections.resize(num_connections);
        for (uint32_t i = 0; i < num_connections; ++i) {
            file >> gene.input_connections[i];
        }
    }
    
    return true;
}

std::string Genome::to_string() const {
    std::stringstream ss;
    ss << "Genome: " << grid_dims_.width << "x" << grid_dims_.height 
       << " I/O: " << num_inputs_ << "/" << num_outputs_ 
       << " Fitness: " << fitness_ << "\n";
    
    auto stats = get_statistics();
    ss << "Active gates: " << stats.active_genes 
       << " Connections: " << stats.total_connections << "\n";
    
    return ss.str();
}

// Private helper methods
void Genome::mutate_gate_type(Gene& gene, std::mt19937& rng) {
    if (gene.is_active) {
        std::uniform_int_distribution<int> gate_type_dist(0, static_cast<int>(GateType::COUNT) - 1);
        GateType new_type;
        do {
            new_type = static_cast<GateType>(gate_type_dist(rng));
        } while (new_type == GateType::NONE);
        
        gene.gate_type = new_type;
    }
}

void Genome::mutate_connections(Gene& gene, std::mt19937& rng) {
    if (gene.is_active && !gene.input_connections.empty()) {
        std::uniform_int_distribution<uint32_t> conn_dist(0, gene.input_connections.size() - 1);
        uint32_t conn_idx = conn_dist(rng);
        
        uint32_t gene_pos = position_to_index(gene.position.x, gene.position.y);
        gene.input_connections[conn_idx] = get_random_valid_connection(rng, gene_pos);
    }
}

void Genome::mutate_activation(Gene& gene, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    if (gene.is_active) {
        // Chance to deactivate
        if (prob_dist(rng) < 0.1f) {
            gene.is_active = false;
            gene.gate_type = GateType::NONE;
            gene.input_connections.clear();
        }
    } else {
        // Chance to activate
        if (prob_dist(rng) < 0.05f) {
            gene.is_active = true;
            gene.gate_type = GateType::AND; // Default gate type
        }
    }
}

void Genome::mutate_position(Gene& gene, std::mt19937& rng) {
    // Position mutation would require restructuring the genome
    // For now, this is a no-op in the grid-based representation
}

uint32_t Genome::position_to_index(uint32_t x, uint32_t y) const {
    return y * grid_dims_.width + x;
}

std::pair<uint32_t, uint32_t> Genome::index_to_position(uint32_t index) const {
    uint32_t y = index / grid_dims_.width;
    uint32_t x = index % grid_dims_.width;
    return {x, y};
}

bool Genome::is_valid_connection(uint32_t from_pos, uint32_t to_pos) const {
    if (from_pos >= genes_.size() || to_pos >= genes_.size()) {
        return false;
    }
    
    // Can't connect to self
    if (from_pos == to_pos) {
        return false;
    }
    
    // For now, allow any connection within the grid
    return true;
}

uint32_t Genome::get_random_valid_connection(std::mt19937& rng, uint32_t gene_position) const {
    std::uniform_int_distribution<uint32_t> pos_dist(0, genes_.size() - 1);
    
    uint32_t connection;
    int attempts = 0;
    do {
        connection = pos_dist(rng);
        attempts++;
    } while (!is_valid_connection(connection, gene_position) && attempts < 100);
    
    return connection;
}

bool Genome::validate_gene(const Gene& gene) const {
    if (gene.is_active) {
        if (gene.gate_type == GateType::NONE) {
            return false;
        }
        
        // Check if gene has correct number of inputs
        uint32_t expected_inputs = get_gate_input_count(gene.gate_type);
        if (gene.input_connections.size() != expected_inputs) {
            return false;
        }
    }
    
    return true;
}

bool Genome::validate_connections() const {
    for (const auto& gene : genes_) {
        if (gene.is_active) {
            for (uint32_t conn : gene.input_connections) {
                if (!is_valid_connection(conn, position_to_index(gene.position.x, gene.position.y))) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool Genome::has_feedback_loops() const {
    // Simple cycle detection using DFS
    std::vector<int> color(genes_.size(), 0); // 0: white, 1: gray, 2: black
    
    std::function<bool(uint32_t)> dfs = [&](uint32_t node) -> bool {
        color[node] = 1; // Mark as gray
        
        for (uint32_t neighbor : genes_[node].output_connections) {
            if (color[neighbor] == 1) {
                return true; // Back edge found (cycle)
            }
            if (color[neighbor] == 0 && dfs(neighbor)) {
                return true;
            }
        }
        
        color[node] = 2; // Mark as black
        return false;
    };
    
    for (uint32_t i = 0; i < genes_.size(); ++i) {
        if (color[i] == 0 && dfs(i)) {
            return true;
        }
    }
    
    return false;
}

void Genome::repair_invalid_connections(std::mt19937& rng) {
    for (auto& gene : genes_) {
        if (gene.is_active) {
            // Remove invalid connections
            gene.input_connections.erase(
                std::remove_if(gene.input_connections.begin(), gene.input_connections.end(),
                    [this, &gene](uint32_t conn) {
                        return !is_valid_connection(conn, position_to_index(gene.position.x, gene.position.y));
                    }),
                gene.input_connections.end()
            );
            
            // Add missing connections
            uint32_t expected_inputs = get_gate_input_count(gene.gate_type);
            while (gene.input_connections.size() < expected_inputs) {
                uint32_t gene_pos = position_to_index(gene.position.x, gene.position.y);
                uint32_t new_conn = get_random_valid_connection(rng, gene_pos);
                gene.input_connections.push_back(new_conn);
            }
        }
    }
}

void Genome::repair_duplicate_positions(std::mt19937& rng) {
    // In grid-based representation, positions are fixed
    // This would be relevant for other representations
}

void Genome::repair_missing_io_gates(std::mt19937& rng) {
    // Ensure we have some active gates if none exist
    bool has_active_gates = std::any_of(genes_.begin(), genes_.end(), 
        [](const Gene& g) { return g.is_active; });
    
    if (!has_active_gates) {
        std::uniform_int_distribution<uint32_t> pos_dist(0, genes_.size() - 1);
        uint32_t idx = pos_dist(rng);
        genes_[idx].is_active = true;
        genes_[idx].gate_type = GateType::AND;
    }
}

// GenomePopulation implementation
GenomePopulation::GenomePopulation(uint32_t population_size, const GridDimensions& grid_dims,
                                 uint32_t num_inputs, uint32_t num_outputs)
    : grid_dims_(grid_dims), num_inputs_(num_inputs), num_outputs_(num_outputs) {
    population_.resize(population_size, Genome(grid_dims, num_inputs, num_outputs));
}

void GenomePopulation::initialize_random(std::mt19937& rng) {
    for (auto& genome : population_) {
        genome.randomize(rng);
    }
}

void GenomePopulation::clear() {
    for (auto& genome : population_) {
        genome.clear();
    }
}

void GenomePopulation::resize(uint32_t new_size) {
    population_.resize(new_size, Genome(grid_dims_, num_inputs_, num_outputs_));
}

float GenomePopulation::get_best_fitness() const {
    if (population_.empty()) return 0.0f;
    
    return std::max_element(population_.begin(), population_.end(),
        [](const Genome& a, const Genome& b) { return a.get_fitness() < b.get_fitness(); })->get_fitness();
}

float GenomePopulation::get_worst_fitness() const {
    if (population_.empty()) return 0.0f;
    
    return std::min_element(population_.begin(), population_.end(),
        [](const Genome& a, const Genome& b) { return a.get_fitness() < b.get_fitness(); })->get_fitness();
}

float GenomePopulation::get_average_fitness() const {
    if (population_.empty()) return 0.0f;
    
    float sum = 0.0f;
    for (const auto& genome : population_) {
        sum += genome.get_fitness();
    }
    return sum / population_.size();
}

const Genome& GenomePopulation::get_best_genome() const {
    return *std::max_element(population_.begin(), population_.end(),
        [](const Genome& a, const Genome& b) { return a.get_fitness() < b.get_fitness(); });
}

const Genome& GenomePopulation::get_worst_genome() const {
    return *std::min_element(population_.begin(), population_.end(),
        [](const Genome& a, const Genome& b) { return a.get_fitness() < b.get_fitness(); });
}

void GenomePopulation::sort_by_fitness() {
    std::sort(population_.begin(), population_.end(),
        [](const Genome& a, const Genome& b) { return a.get_fitness() > b.get_fitness(); });
}

// Utility functions
std::unique_ptr<Genome> create_random_genome(const GridDimensions& grid_dims,
                                           uint32_t num_inputs, uint32_t num_outputs,
                                           std::mt19937& rng) {
    auto genome = std::make_unique<Genome>(grid_dims, num_inputs, num_outputs);
    genome->randomize(rng);
    return genome;
}

std::unique_ptr<Genome> create_minimal_genome(const GridDimensions& grid_dims,
                                            uint32_t num_inputs, uint32_t num_outputs) {
    auto genome = std::make_unique<Genome>(grid_dims, num_inputs, num_outputs);
    // Minimal genome with just a few active gates
    auto& genes = genome->get_genes();
    if (!genes.empty()) {
        genes[0].is_active = true;
        genes[0].gate_type = GateType::AND;
    }
    return genome;
}

float calculate_genome_similarity(const Genome& genome1, const Genome& genome2) {
    if (genome1.get_gene_count() != genome2.get_gene_count()) {
        return 0.0f;
    }
    
    uint32_t matches = 0;
    for (uint32_t i = 0; i < genome1.get_gene_count(); ++i) {
        const auto& gene1 = genome1.get_gene(i);
        const auto& gene2 = genome2.get_gene(i);
        
        if (gene1.is_active == gene2.is_active && 
            gene1.gate_type == gene2.gate_type &&
            gene1.input_connections == gene2.input_connections) {
            matches++;
        }
    }
    
    return static_cast<float>(matches) / genome1.get_gene_count();
}

} // namespace circuit 