#include "circuit/ga/genetic_algorithm.h"
#include "circuit/core/types.h"
#include <algorithm>
#include <random>
#include <vector>

namespace circuit {

// Basic gene mutation
void mutate_gene(Gene& gene, const GridDimensions& grid_dims, 
                uint32_t num_inputs, uint32_t num_outputs, 
                std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    // Mutation probabilities
    const float ACTIVATION_MUTATION_RATE = 0.1f;
    const float GATE_TYPE_MUTATION_RATE = 0.2f;
    const float CONNECTION_MUTATION_RATE = 0.3f;
    
    // Activation mutation
    if (prob_dist(rng) < ACTIVATION_MUTATION_RATE) {
        gene.is_active = !gene.is_active;
        
        if (gene.is_active && gene.gate_type == GateType::NONE) {
            // Assign a random gate type
            std::uniform_int_distribution<int> gate_dist(0, static_cast<int>(GateType::COUNT) - 1);
            do {
                gene.gate_type = static_cast<GateType>(gate_dist(rng));
            } while (gene.gate_type == GateType::NONE);
        } else if (!gene.is_active) {
            gene.gate_type = GateType::NONE;
            gene.input_connections.clear();
        }
    }
    
    // Gate type mutation
    if (gene.is_active && prob_dist(rng) < GATE_TYPE_MUTATION_RATE) {
        std::uniform_int_distribution<int> gate_dist(0, static_cast<int>(GateType::COUNT) - 1);
        GateType new_type;
        do {
            new_type = static_cast<GateType>(gate_dist(rng));
        } while (new_type == GateType::NONE);
        
        gene.gate_type = new_type;
        
        // Adjust connections for new gate type
        uint32_t required_inputs = get_gate_input_count(new_type);
        if (gene.input_connections.size() != required_inputs) {
            gene.input_connections.resize(required_inputs);
            
            // Fill with random connections
            std::uniform_int_distribution<uint32_t> conn_dist(0, grid_dims.width * grid_dims.height - 1);
            for (uint32_t& conn : gene.input_connections) {
                conn = conn_dist(rng);
            }
        }
    }
    
    // Connection mutation
    if (gene.is_active && prob_dist(rng) < CONNECTION_MUTATION_RATE) {
        if (!gene.input_connections.empty()) {
            std::uniform_int_distribution<uint32_t> conn_idx_dist(0, gene.input_connections.size() - 1);
            std::uniform_int_distribution<uint32_t> conn_dist(0, grid_dims.width * grid_dims.height - 1);
            
            uint32_t idx = conn_idx_dist(rng);
            gene.input_connections[idx] = conn_dist(rng);
        }
    }
}

// Gaussian mutation for numerical parameters
void gaussian_mutation(Genome& genome, float mutation_rate, float sigma, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::normal_distribution<float> gauss_dist(0.0f, sigma);
    
    // For circuit genomes, apply gaussian noise to connection strengths
    // This is a conceptual implementation - actual circuits have discrete connections
    
    for (uint32_t i = 0; i < genome.get_gene_count(); ++i) {
        if (prob_dist(rng) < mutation_rate) {
            Gene& gene = genome.get_genes()[i];
            
            if (gene.is_active) {
                // Add gaussian noise to connection selection
                for (uint32_t& conn : gene.input_connections) {
                    float noise = gauss_dist(rng);
                    int32_t new_conn = static_cast<int32_t>(conn) + static_cast<int32_t>(noise);
                    
                    // Clamp to valid range
                    new_conn = std::max(0, std::min(new_conn, static_cast<int32_t>(genome.get_gene_count() - 1)));
                    conn = static_cast<uint32_t>(new_conn);
                }
            }
        }
    }
}

// Uniform mutation
void uniform_mutation(Genome& genome, float mutation_rate, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (uint32_t i = 0; i < genome.get_gene_count(); ++i) {
        if (prob_dist(rng) < mutation_rate) {
            Gene& gene = genome.get_genes()[i];
            mutate_gene(gene, genome.get_grid_dimensions(), 
                       genome.get_num_inputs(), genome.get_num_outputs(), rng);
        }
    }
}

// Boundary mutation - keeps mutations within valid bounds
void boundary_mutation(Genome& genome, float mutation_rate, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (uint32_t i = 0; i < genome.get_gene_count(); ++i) {
        if (prob_dist(rng) < mutation_rate) {
            Gene& gene = genome.get_genes()[i];
            
            if (gene.is_active) {
                // Mutate connections to boundary values
                for (uint32_t& conn : gene.input_connections) {
                    if (prob_dist(rng) < 0.5f) {
                        conn = 0; // Minimum boundary
                    } else {
                        conn = genome.get_gene_count() - 1; // Maximum boundary
                    }
                }
            }
        }
    }
}

// Polynomial mutation
void polynomial_mutation(Genome& genome, float mutation_rate, float eta, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (uint32_t i = 0; i < genome.get_gene_count(); ++i) {
        if (prob_dist(rng) < mutation_rate) {
            Gene& gene = genome.get_genes()[i];
            
            if (gene.is_active) {
                // Apply polynomial mutation to connections
                for (uint32_t& conn : gene.input_connections) {
                    float u = prob_dist(rng);
                    float delta;
                    
                    if (u < 0.5f) {
                        delta = std::pow(2.0f * u, 1.0f / (eta + 1.0f)) - 1.0f;
                    } else {
                        delta = 1.0f - std::pow(2.0f * (1.0f - u), 1.0f / (eta + 1.0f));
                    }
                    
                    // Apply mutation
                    float normalized_conn = static_cast<float>(conn) / (genome.get_gene_count() - 1);
                    normalized_conn += delta;
                    normalized_conn = std::max(0.0f, std::min(1.0f, normalized_conn));
                    
                    conn = static_cast<uint32_t>(normalized_conn * (genome.get_gene_count() - 1));
                }
            }
        }
    }
}

// Swap mutation - swaps two genes
void swap_mutation(Genome& genome, float mutation_rate, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    if (prob_dist(rng) < mutation_rate) {
        std::uniform_int_distribution<uint32_t> gene_dist(0, genome.get_gene_count() - 1);
        
        uint32_t idx1 = gene_dist(rng);
        uint32_t idx2 = gene_dist(rng);
        
        if (idx1 != idx2) {
            // Swap the genes (but keep positions)
            Gene temp = genome.get_genes()[idx1];
            Position pos1 = temp.position;
            Position pos2 = genome.get_genes()[idx2].position;
            
            genome.get_genes()[idx1] = genome.get_genes()[idx2];
            genome.get_genes()[idx2] = temp;
            
            // Restore positions
            genome.get_genes()[idx1].position = pos1;
            genome.get_genes()[idx2].position = pos2;
        }
    }
}

// Inversion mutation - reverses a segment of genes
void inversion_mutation(Genome& genome, float mutation_rate, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    if (prob_dist(rng) < mutation_rate) {
        std::uniform_int_distribution<uint32_t> gene_dist(0, genome.get_gene_count() - 1);
        
        uint32_t start = gene_dist(rng);
        uint32_t end = gene_dist(rng);
        
        if (start > end) {
            std::swap(start, end);
        }
        
        if (start != end) {
            // Reverse the segment (preserving positions)
            std::vector<Gene> segment;
            for (uint32_t i = start; i <= end; ++i) {
                segment.push_back(genome.get_genes()[i]);
            }
            
            std::reverse(segment.begin(), segment.end());
            
            // Restore positions
            for (uint32_t i = 0; i < segment.size(); ++i) {
                segment[i].position = genome.get_genes()[start + i].position;
            }
            
            // Copy back
            for (uint32_t i = 0; i < segment.size(); ++i) {
                genome.get_genes()[start + i] = segment[i];
            }
        }
    }
}

// Insertion mutation - moves a gene to a new position
void insertion_mutation(Genome& genome, float mutation_rate, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    if (prob_dist(rng) < mutation_rate) {
        std::uniform_int_distribution<uint32_t> gene_dist(0, genome.get_gene_count() - 1);
        
        uint32_t source = gene_dist(rng);
        uint32_t target = gene_dist(rng);
        
        if (source != target) {
            Gene temp = genome.get_genes()[source];
            Position target_pos = genome.get_genes()[target].position;
            
            // Shift genes
            if (source < target) {
                for (uint32_t i = source; i < target; ++i) {
                    genome.get_genes()[i] = genome.get_genes()[i + 1];
                }
            } else {
                for (uint32_t i = source; i > target; --i) {
                    genome.get_genes()[i] = genome.get_genes()[i - 1];
                }
            }
            
            // Insert at target position
            genome.get_genes()[target] = temp;
            genome.get_genes()[target].position = target_pos;
        }
    }
}

// Scramble mutation - randomly shuffles a segment
void scramble_mutation(Genome& genome, float mutation_rate, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    if (prob_dist(rng) < mutation_rate) {
        std::uniform_int_distribution<uint32_t> gene_dist(0, genome.get_gene_count() - 1);
        
        uint32_t start = gene_dist(rng);
        uint32_t end = gene_dist(rng);
        
        if (start > end) {
            std::swap(start, end);
        }
        
        if (start != end) {
            // Extract segment
            std::vector<Gene> segment;
            std::vector<Position> positions;
            
            for (uint32_t i = start; i <= end; ++i) {
                segment.push_back(genome.get_genes()[i]);
                positions.push_back(genome.get_genes()[i].position);
            }
            
            // Shuffle segment
            std::shuffle(segment.begin(), segment.end(), rng);
            
            // Restore positions
            for (uint32_t i = 0; i < segment.size(); ++i) {
                segment[i].position = positions[i];
            }
            
            // Copy back
            for (uint32_t i = 0; i < segment.size(); ++i) {
                genome.get_genes()[start + i] = segment[i];
            }
        }
    }
}

// Adaptive mutation - adjusts mutation rate based on fitness
void adaptive_mutation(Genome& genome, float base_mutation_rate, 
                      float fitness_threshold, std::mt19937& rng) {
    float adjusted_rate = base_mutation_rate;
    
    // Increase mutation rate for low-fitness genomes
    if (genome.get_fitness() < fitness_threshold) {
        adjusted_rate *= 2.0f;
    }
    
    // Apply standard mutation with adjusted rate
    uniform_mutation(genome, adjusted_rate, rng);
}

// Multi-level mutation - applies different mutation operators
void multi_level_mutation(Genome& genome, const MutationConfig& config, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (uint32_t i = 0; i < genome.get_gene_count(); ++i) {
        Gene& gene = genome.get_genes()[i];
        
        // Gate type mutation
        if (gene.is_active && prob_dist(rng) < config.gate_type_mutation_rate) {
            std::uniform_int_distribution<int> gate_dist(0, static_cast<int>(GateType::COUNT) - 1);
            GateType new_type;
            do {
                new_type = static_cast<GateType>(gate_dist(rng));
            } while (new_type == GateType::NONE);
            
            gene.gate_type = new_type;
        }
        
        // Connection mutation
        if (gene.is_active && prob_dist(rng) < config.connection_mutation_rate) {
            if (!gene.input_connections.empty()) {
                std::uniform_int_distribution<uint32_t> conn_idx_dist(0, gene.input_connections.size() - 1);
                std::uniform_int_distribution<uint32_t> conn_dist(0, genome.get_gene_count() - 1);
                
                uint32_t idx = conn_idx_dist(rng);
                gene.input_connections[idx] = conn_dist(rng);
            }
        }
        
        // Activation mutation
        if (prob_dist(rng) < config.activation_mutation_rate) {
            gene.is_active = !gene.is_active;
            
            if (gene.is_active && gene.gate_type == GateType::NONE) {
                gene.gate_type = GateType::AND; // Default gate type
            } else if (!gene.is_active) {
                gene.gate_type = GateType::NONE;
                gene.input_connections.clear();
            }
        }
    }
    
    // Structural mutations
    if (prob_dist(rng) < config.add_gate_rate) {
        // Add a new active gate
        std::uniform_int_distribution<uint32_t> gene_dist(0, genome.get_gene_count() - 1);
        uint32_t idx = gene_dist(rng);
        
        Gene& gene = genome.get_genes()[idx];
        if (!gene.is_active) {
            gene.is_active = true;
            gene.gate_type = GateType::AND;
            
            // Add random connections
            uint32_t required_inputs = get_gate_input_count(gene.gate_type);
            gene.input_connections.clear();
            
            std::uniform_int_distribution<uint32_t> conn_dist(0, genome.get_gene_count() - 1);
            for (uint32_t j = 0; j < required_inputs; ++j) {
                gene.input_connections.push_back(conn_dist(rng));
            }
        }
    }
    
    if (prob_dist(rng) < config.remove_gate_rate) {
        // Remove an active gate
        std::vector<uint32_t> active_indices;
        for (uint32_t i = 0; i < genome.get_gene_count(); ++i) {
            if (genome.get_genes()[i].is_active) {
                active_indices.push_back(i);
            }
        }
        
        if (!active_indices.empty()) {
            std::uniform_int_distribution<uint32_t> active_dist(0, active_indices.size() - 1);
            uint32_t idx = active_indices[active_dist(rng)];
            
            Gene& gene = genome.get_genes()[idx];
            gene.is_active = false;
            gene.gate_type = GateType::NONE;
            gene.input_connections.clear();
        }
    }
}

// Mutation factory function
void perform_mutation(Genome& genome, float mutation_rate, std::mt19937& rng) {
    // Use uniform mutation as the default
    uniform_mutation(genome, mutation_rate, rng);
}

// Self-adaptive mutation
void self_adaptive_mutation(Genome& genome, float& mutation_rate, 
                           float tau, std::mt19937& rng) {
    std::normal_distribution<float> gauss_dist(0.0f, 1.0f);
    
    // Adapt mutation rate
    mutation_rate *= std::exp(tau * gauss_dist(rng));
    
    // Clamp mutation rate
    mutation_rate = std::max(0.001f, std::min(0.5f, mutation_rate));
    
    // Apply mutation
    uniform_mutation(genome, mutation_rate, rng);
}

// Directed mutation - biased towards better solutions
void directed_mutation(Genome& genome, const Genome& target, 
                      float mutation_rate, float bias, std::mt19937& rng) {
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (uint32_t i = 0; i < genome.get_gene_count(); ++i) {
        if (prob_dist(rng) < mutation_rate) {
            Gene& gene = genome.get_genes()[i];
            const Gene& target_gene = target.get_gene(i);
            
            // Bias mutation towards target
            if (prob_dist(rng) < bias) {
                // Move towards target
                if (target_gene.is_active && !gene.is_active) {
                    gene.is_active = true;
                    gene.gate_type = target_gene.gate_type;
                    gene.input_connections = target_gene.input_connections;
                } else if (!target_gene.is_active && gene.is_active) {
                    gene.is_active = false;
                    gene.gate_type = GateType::NONE;
                    gene.input_connections.clear();
                } else if (target_gene.is_active && gene.is_active) {
                    gene.gate_type = target_gene.gate_type;
                    // Blend connections
                    for (uint32_t j = 0; j < std::min(gene.input_connections.size(), 
                                                     target_gene.input_connections.size()); ++j) {
                        if (prob_dist(rng) < 0.5f) {
                            gene.input_connections[j] = target_gene.input_connections[j];
                        }
                    }
                }
            } else {
                // Random mutation
                mutate_gene(gene, genome.get_grid_dimensions(), 
                           genome.get_num_inputs(), genome.get_num_outputs(), rng);
            }
        }
    }
}

} // namespace circuit 