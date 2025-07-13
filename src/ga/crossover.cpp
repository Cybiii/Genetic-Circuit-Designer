#include "circuit/ga/genetic_algorithm.h"
#include "circuit/core/types.h"
#include "circuit/utils/utils.h"
#include <cassert>
#include <algorithm>
#include <random>
#include <cmath>
#include <vector>

namespace circuit {

// Single point crossover
std::pair<Genome, Genome> single_point_crossover(const Genome& parent1, 
                                                 const Genome& parent2, 
                                                 std::mt19937& rng) {
    assert(parent1.get_gene_count() == parent2.get_gene_count());
    
    Genome offspring1(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    Genome offspring2(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    
    // Choose crossover point
    std::uniform_int_distribution<uint32_t> point_dist(1, parent1.get_gene_count() - 1);
    uint32_t crossover_point = point_dist(rng);
    
    // Copy genes
    for (uint32_t i = 0; i < offspring1.get_gene_count(); ++i) {
        if (i < crossover_point) {
            offspring1.get_genes()[i] = parent1.get_gene(i);
            offspring2.get_genes()[i] = parent2.get_gene(i);
        } else {
            offspring1.get_genes()[i] = parent2.get_gene(i);
            offspring2.get_genes()[i] = parent1.get_gene(i);
        }
    }
    
    // Repair offspring
    std::mt19937 repair_rng(rng());
    offspring1.repair(repair_rng);
    offspring2.repair(repair_rng);
    
    return std::make_pair(std::move(offspring1), std::move(offspring2));
}

// Two point crossover
std::pair<Genome, Genome> two_point_crossover(const Genome& parent1, 
                                             const Genome& parent2, 
                                             std::mt19937& rng) {
    assert(parent1.get_gene_count() == parent2.get_gene_count());
    
    Genome offspring1(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    Genome offspring2(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    
    // Choose two crossover points
    std::uniform_int_distribution<uint32_t> point_dist(0, parent1.get_gene_count() - 1);
    uint32_t point1 = point_dist(rng);
    uint32_t point2 = point_dist(rng);
    
    if (point1 > point2) {
        std::swap(point1, point2);
    }
    
    // Copy genes
    for (uint32_t i = 0; i < offspring1.get_gene_count(); ++i) {
        if (i < point1 || i > point2) {
            offspring1.get_genes()[i] = parent1.get_gene(i);
            offspring2.get_genes()[i] = parent2.get_gene(i);
        } else {
            offspring1.get_genes()[i] = parent2.get_gene(i);
            offspring2.get_genes()[i] = parent1.get_gene(i);
        }
    }
    
    // Repair offspring
    std::mt19937 repair_rng(rng());
    offspring1.repair(repair_rng);
    offspring2.repair(repair_rng);
    
    return std::make_pair(std::move(offspring1), std::move(offspring2));
}

// Uniform crossover
std::pair<Genome, Genome> uniform_crossover(const Genome& parent1, 
                                           const Genome& parent2, 
                                           std::mt19937& rng) {
    assert(parent1.get_gene_count() == parent2.get_gene_count());
    
    Genome offspring1(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    Genome offspring2(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    // For each gene, randomly choose which parent to inherit from
    for (uint32_t i = 0; i < offspring1.get_gene_count(); ++i) {
        if (prob_dist(rng) < 0.5f) {
            offspring1.get_genes()[i] = parent1.get_gene(i);
            offspring2.get_genes()[i] = parent2.get_gene(i);
        } else {
            offspring1.get_genes()[i] = parent2.get_gene(i);
            offspring2.get_genes()[i] = parent1.get_gene(i);
        }
    }
    
    // Repair offspring
    std::mt19937 repair_rng(rng());
    offspring1.repair(repair_rng);
    offspring2.repair(repair_rng);
    
    return std::make_pair(std::move(offspring1), std::move(offspring2));
}

// Grid-based crossover (specialized for circuit layout)
std::pair<Genome, Genome> grid_based_crossover(const Genome& parent1, 
                                              const Genome& parent2, 
                                              std::mt19937& rng) {
    assert(parent1.get_gene_count() == parent2.get_gene_count());
    
    Genome offspring1(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    Genome offspring2(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    
    const GridDimensions& grid = parent1.get_grid_dimensions();
    
    // Choose crossover strategy
    std::uniform_int_distribution<int> strategy_dist(0, 2);
    int strategy = strategy_dist(rng);
    
    switch (strategy) {
        case 0: // Vertical split
        {
            std::uniform_int_distribution<uint32_t> split_dist(1, grid.width - 1);
            uint32_t split_x = split_dist(rng);
            
            for (uint32_t y = 0; y < grid.height; ++y) {
                for (uint32_t x = 0; x < grid.width; ++x) {
                    uint32_t idx = y * grid.width + x;
                    
                    if (x < split_x) {
                        offspring1.get_genes()[idx] = parent1.get_gene(idx);
                        offspring2.get_genes()[idx] = parent2.get_gene(idx);
                    } else {
                        offspring1.get_genes()[idx] = parent2.get_gene(idx);
                        offspring2.get_genes()[idx] = parent1.get_gene(idx);
                    }
                }
            }
            break;
        }
        case 1: // Horizontal split
        {
            std::uniform_int_distribution<uint32_t> split_dist(1, grid.height - 1);
            uint32_t split_y = split_dist(rng);
            
            for (uint32_t y = 0; y < grid.height; ++y) {
                for (uint32_t x = 0; x < grid.width; ++x) {
                    uint32_t idx = y * grid.width + x;
                    
                    if (y < split_y) {
                        offspring1.get_genes()[idx] = parent1.get_gene(idx);
                        offspring2.get_genes()[idx] = parent2.get_gene(idx);
                    } else {
                        offspring1.get_genes()[idx] = parent2.get_gene(idx);
                        offspring2.get_genes()[idx] = parent1.get_gene(idx);
                    }
                }
            }
            break;
        }
        case 2: // Quadrant-based
        {
            std::uniform_int_distribution<uint32_t> quad_dist(0, 3);
            uint32_t quad1 = quad_dist(rng);
            uint32_t quad2 = quad_dist(rng);
            
            uint32_t mid_x = grid.width / 2;
            uint32_t mid_y = grid.height / 2;
            
            for (uint32_t y = 0; y < grid.height; ++y) {
                for (uint32_t x = 0; x < grid.width; ++x) {
                    uint32_t idx = y * grid.width + x;
                    
                    // Determine quadrant
                    uint32_t quadrant = 0;
                    if (x >= mid_x && y < mid_y) quadrant = 1;
                    else if (x >= mid_x && y >= mid_y) quadrant = 2;
                    else if (x < mid_x && y >= mid_y) quadrant = 3;
                    
                    if (quadrant == quad1 || quadrant == quad2) {
                        offspring1.get_genes()[idx] = parent1.get_gene(idx);
                        offspring2.get_genes()[idx] = parent2.get_gene(idx);
                    } else {
                        offspring1.get_genes()[idx] = parent2.get_gene(idx);
                        offspring2.get_genes()[idx] = parent1.get_gene(idx);
                    }
                }
            }
            break;
        }
    }
    
    // Repair offspring
    std::mt19937 repair_rng(rng());
    offspring1.repair(repair_rng);
    offspring2.repair(repair_rng);
    
    return std::make_pair(std::move(offspring1), std::move(offspring2));
}

// Arithmetic crossover (for numerical gene values)
std::pair<Genome, Genome> arithmetic_crossover(const Genome& parent1, 
                                              const Genome& parent2, 
                                              float alpha,
                                              std::mt19937& rng) {
    assert(parent1.get_gene_count() == parent2.get_gene_count());
    
    Genome offspring1(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    Genome offspring2(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    
    // For circuit genomes, arithmetic crossover is not directly applicable
    // We'll use a modified approach where we blend connection weights
    
    std::uniform_real_distribution<float> blend_dist(0.0f, 1.0f);
    
    for (uint32_t i = 0; i < offspring1.get_gene_count(); ++i) {
        const Gene& gene1 = parent1.get_gene(i);
        const Gene& gene2 = parent2.get_gene(i);
        
        // Blend activation states
        float activation_blend = blend_dist(rng);
        bool use_parent1 = activation_blend < alpha;
        
        offspring1.get_genes()[i] = use_parent1 ? gene1 : gene2;
        offspring2.get_genes()[i] = use_parent1 ? gene2 : gene1;
        
        // For connections, we can try to blend them
        if (gene1.is_active && gene2.is_active && gene1.gate_type == gene2.gate_type) {
            // If both genes are active and same type, blend their connections
            Gene& off1_gene = offspring1.get_genes()[i];
            Gene& off2_gene = offspring2.get_genes()[i];
            
            // Randomly select connections from both parents
            std::vector<ConnectionId> combined_connections = gene1.input_connections;
            combined_connections.insert(combined_connections.end(), 
                                      gene2.input_connections.begin(), 
                                      gene2.input_connections.end());
            
            // Remove duplicates
            std::sort(combined_connections.begin(), combined_connections.end());
            combined_connections.erase(std::unique(combined_connections.begin(), 
                                                 combined_connections.end()), 
                                     combined_connections.end());
            
            // Randomly select subset for each offspring
            std::shuffle(combined_connections.begin(), combined_connections.end(), rng);
            
            uint32_t required_connections = get_gate_input_count(gene1.gate_type);
            if (combined_connections.size() >= required_connections) {
                off1_gene.input_connections.assign(combined_connections.begin(), 
                                                  combined_connections.begin() + required_connections);
                
                // For second offspring, rotate the selection
                std::rotate(combined_connections.begin(), 
                           combined_connections.begin() + 1, 
                           combined_connections.end());
                
                off2_gene.input_connections.assign(combined_connections.begin(), 
                                                  combined_connections.begin() + required_connections);
            }
        }
    }
    
    // Repair offspring
    std::mt19937 repair_rng(rng());
    offspring1.repair(repair_rng);
    offspring2.repair(repair_rng);
    
    return std::make_pair(std::move(offspring1), std::move(offspring2));
}

// Partially matched crossover (PMX) - adapted for circuit genomes
std::pair<Genome, Genome> partially_matched_crossover(const Genome& parent1, 
                                                     const Genome& parent2, 
                                                     std::mt19937& rng) {
    assert(parent1.get_gene_count() == parent2.get_gene_count());
    
    Genome offspring1(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    Genome offspring2(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    
    // Initialize with parents
    for (uint32_t i = 0; i < offspring1.get_gene_count(); ++i) {
        offspring1.get_genes()[i] = parent1.get_gene(i);
        offspring2.get_genes()[i] = parent2.get_gene(i);
    }
    
    // Choose crossover segment
    std::uniform_int_distribution<uint32_t> point_dist(0, parent1.get_gene_count() - 1);
    uint32_t start = point_dist(rng);
    uint32_t end = point_dist(rng);
    
    if (start > end) {
        std::swap(start, end);
    }
    
    // Create mapping of active positions in the crossover segment
    std::vector<std::pair<uint32_t, uint32_t>> position_mapping;
    
    for (uint32_t i = start; i <= end; ++i) {
        if (parent1.get_gene(i).is_active && parent2.get_gene(i).is_active) {
            // Swap active genes in the segment
            offspring1.get_genes()[i] = parent2.get_gene(i);
            offspring2.get_genes()[i] = parent1.get_gene(i);
            
            // Record position mapping for connection repair
            position_mapping.emplace_back(i, i);
        }
    }
    
    // Repair connections that point to swapped positions
    for (uint32_t i = 0; i < offspring1.get_gene_count(); ++i) {
        if (i >= start && i <= end) continue; // Skip the crossover segment
        
        Gene& gene1 = offspring1.get_genes()[i];
        Gene& gene2 = offspring2.get_genes()[i];
        
        // Update connections based on position mapping
        for (auto& conn : gene1.input_connections) {
            for (const auto& mapping : position_mapping) {
                if (conn == mapping.first) {
                    conn = mapping.second;
                    break;
                }
            }
        }
        
        for (auto& conn : gene2.input_connections) {
            for (const auto& mapping : position_mapping) {
                if (conn == mapping.first) {
                    conn = mapping.second;
                    break;
                }
            }
        }
    }
    
    // Repair offspring
    std::mt19937 repair_rng(rng());
    offspring1.repair(repair_rng);
    offspring2.repair(repair_rng);
    
    return std::make_pair(std::move(offspring1), std::move(offspring2));
}

// Simulated binary crossover (SBX) - adapted for circuit parameters
std::pair<Genome, Genome> simulated_binary_crossover(const Genome& parent1, 
                                                    const Genome& parent2, 
                                                    float eta,
                                                    std::mt19937& rng) {
    assert(parent1.get_gene_count() == parent2.get_gene_count());
    
    Genome offspring1(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    Genome offspring2(parent1.get_grid_dimensions(), parent1.get_num_inputs(), parent1.get_num_outputs());
    
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (uint32_t i = 0; i < offspring1.get_gene_count(); ++i) {
        const Gene& gene1 = parent1.get_gene(i);
        const Gene& gene2 = parent2.get_gene(i);
        
        // For binary traits (activation), use standard crossover
        if (prob_dist(rng) < 0.5f) {
            offspring1.get_genes()[i] = gene1;
            offspring2.get_genes()[i] = gene2;
        } else {
            offspring1.get_genes()[i] = gene2;
            offspring2.get_genes()[i] = gene1;
        }
        
        // For connections, apply SBX-like blending
        if (gene1.is_active && gene2.is_active && gene1.gate_type == gene2.gate_type) {
            Gene& off1_gene = offspring1.get_genes()[i];
            Gene& off2_gene = offspring2.get_genes()[i];
            
            // Blend connection patterns
            float u = prob_dist(rng);
            float beta;
            
            if (u <= 0.5f) {
                beta = std::pow(2.0f * u, 1.0f / (eta + 1.0f));
            } else {
                beta = std::pow(1.0f / (2.0f * (1.0f - u)), 1.0f / (eta + 1.0f));
            }
            
            // Apply blending to connection selection
            // This is an approximation for discrete connection values
            if (beta > 1.0f) {
                // Favor parent 1 connections
                off1_gene.input_connections = gene1.input_connections;
                off2_gene.input_connections = gene2.input_connections;
            } else {
                // Favor parent 2 connections
                off1_gene.input_connections = gene2.input_connections;
                off2_gene.input_connections = gene1.input_connections;
            }
        }
    }
    
    // Repair offspring
    std::mt19937 repair_rng(rng());
    offspring1.repair(repair_rng);
    offspring2.repair(repair_rng);
    
    return std::make_pair(std::move(offspring1), std::move(offspring2));
}

// Crossover factory function
std::pair<Genome, Genome> perform_crossover(const Genome& parent1, 
                                           const Genome& parent2, 
                                           CrossoverType type,
                                           std::mt19937& rng) {
    switch (type) {
        case CrossoverType::SINGLE_POINT:
            return single_point_crossover(parent1, parent2, rng);
        case CrossoverType::TWO_POINT:
            return two_point_crossover(parent1, parent2, rng);
        case CrossoverType::UNIFORM:
            return uniform_crossover(parent1, parent2, rng);
        case CrossoverType::GRID_BASED:
            return grid_based_crossover(parent1, parent2, rng);
        default:
            return two_point_crossover(parent1, parent2, rng);
    }
}

// Adaptive crossover rate
float calculate_adaptive_crossover_rate(const GenomePopulation& population, 
                                       float base_rate,
                                       float diversity_threshold) {
    float diversity = 0.0f;
    uint32_t comparisons = 0;
    
    // Calculate average diversity
    for (uint32_t i = 0; i < population.size(); ++i) {
        for (uint32_t j = i + 1; j < population.size(); ++j) {
            float similarity = calculate_genome_similarity(population[i], population[j]);
            diversity += (1.0f - similarity);
            comparisons++;
        }
    }
    
    if (comparisons > 0) {
        diversity /= comparisons;
    }
    
    // Adjust crossover rate based on diversity
    if (diversity < diversity_threshold) {
        // Low diversity - increase crossover rate
        return std::min(0.95f, base_rate * 1.2f);
    } else {
        // High diversity - decrease crossover rate
        return std::max(0.1f, base_rate * 0.8f);
    }
}

} // namespace circuit 