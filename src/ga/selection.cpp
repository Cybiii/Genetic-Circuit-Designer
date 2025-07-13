#include "circuit/ga/genetic_algorithm.h"
#include "circuit/core/types.h"
#include <algorithm>
#include <random>
#include <vector>

namespace circuit {

// Tournament selection implementation
std::vector<uint32_t> tournament_selection(const GenomePopulation& population,
                                          uint32_t num_parents,
                                          uint32_t tournament_size,
                                          std::mt19937& rng) {
    std::vector<uint32_t> selected;
    selected.reserve(num_parents);
    
    std::uniform_int_distribution<uint32_t> pop_dist(0, population.size() - 1);
    
    for (uint32_t p = 0; p < num_parents; ++p) {
        // Tournament selection
        uint32_t best_idx = pop_dist(rng);
        float best_fitness = population[best_idx].get_fitness();
        
        for (uint32_t i = 1; i < tournament_size; ++i) {
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

// Roulette wheel selection implementation
std::vector<uint32_t> roulette_wheel_selection(const GenomePopulation& population,
                                             uint32_t num_parents,
                                             std::mt19937& rng) {
    std::vector<uint32_t> selected;
    selected.reserve(num_parents);
    
    // Calculate fitness sum (ensure all fitnesses are positive)
    float total_fitness = 0.0f;
    float min_fitness = population.get_worst_fitness();
    float offset = min_fitness < 0.0f ? -min_fitness + 1.0f : 0.0f;
    
    for (uint32_t i = 0; i < population.size(); ++i) {
        total_fitness += population[i].get_fitness() + offset;
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
            current_sum += population[i].get_fitness() + offset;
            if (current_sum >= wheel_pos) {
                selected.push_back(i);
                break;
            }
        }
    }
    
    return selected;
}

// Rank-based selection implementation
std::vector<uint32_t> rank_based_selection(const GenomePopulation& population,
                                         uint32_t num_parents,
                                         std::mt19937& rng) {
    std::vector<uint32_t> selected;
    selected.reserve(num_parents);
    
    // Create ranking
    std::vector<std::pair<float, uint32_t>> fitness_rank;
    for (uint32_t i = 0; i < population.size(); ++i) {
        fitness_rank.emplace_back(population[i].get_fitness(), i);
    }
    
    // Sort by fitness (descending)
    std::sort(fitness_rank.begin(), fitness_rank.end(), 
              [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
                  return a.first > b.first;
              });
    
    // Linear ranking: rank 1 gets weight N, rank 2 gets weight N-1, etc.
    float total_rank = population.size() * (population.size() + 1) / 2.0f;
    std::uniform_real_distribution<float> rank_dist(0.0f, total_rank);
    
    for (uint32_t p = 0; p < num_parents; ++p) {
        float rank_pos = rank_dist(rng);
        float current_sum = 0.0f;
        
        for (uint32_t i = 0; i < population.size(); ++i) {
            current_sum += (population.size() - i); // Higher rank for better fitness
            if (current_sum >= rank_pos) {
                selected.push_back(fitness_rank[i].second);
                break;
            }
        }
    }
    
    return selected;
}

// Elitist selection implementation
std::vector<uint32_t> elitist_selection(const GenomePopulation& population,
                                       uint32_t num_parents,
                                       uint32_t elite_count,
                                       std::mt19937& rng) {
    std::vector<uint32_t> selected;
    selected.reserve(num_parents);
    
    // Create sorted indices by fitness
    std::vector<std::pair<float, uint32_t>> fitness_indices;
    for (uint32_t i = 0; i < population.size(); ++i) {
        fitness_indices.emplace_back(population[i].get_fitness(), i);
    }
    
    // Sort by fitness (descending)
    std::sort(fitness_indices.begin(), fitness_indices.end(), 
              [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
                  return a.first > b.first;
              });
    
    // Select from top performers
    uint32_t elite_pool_size = std::min(elite_count * 2, static_cast<uint32_t>(population.size()));
    std::uniform_int_distribution<uint32_t> elite_dist(0, elite_pool_size - 1);
    
    for (uint32_t p = 0; p < num_parents; ++p) {
        uint32_t elite_idx = elite_dist(rng);
        selected.push_back(fitness_indices[elite_idx].second);
    }
    
    return selected;
}

// Stochastic universal sampling
std::vector<uint32_t> stochastic_universal_sampling(const GenomePopulation& population,
                                                   uint32_t num_parents,
                                                   std::mt19937& rng) {
    std::vector<uint32_t> selected;
    selected.reserve(num_parents);
    
    // Calculate fitness sum
    float total_fitness = 0.0f;
    float min_fitness = population.get_worst_fitness();
    float offset = min_fitness < 0.0f ? -min_fitness + 1.0f : 0.0f;
    
    for (uint32_t i = 0; i < population.size(); ++i) {
        total_fitness += population[i].get_fitness() + offset;
    }
    
    if (total_fitness <= 0.0f) {
        // Fallback to random selection
        std::uniform_int_distribution<uint32_t> pop_dist(0, population.size() - 1);
        for (uint32_t p = 0; p < num_parents; ++p) {
            selected.push_back(pop_dist(rng));
        }
        return selected;
    }
    
    // Calculate pointer distance
    float pointer_distance = total_fitness / num_parents;
    
    // Random starting point
    std::uniform_real_distribution<float> start_dist(0.0f, pointer_distance);
    float start_point = start_dist(rng);
    
    // Select individuals
    float current_sum = 0.0f;
    uint32_t current_individual = 0;
    
    for (uint32_t p = 0; p < num_parents; ++p) {
        float pointer = start_point + p * pointer_distance;
        
        // Find the individual at this pointer position
        while (current_sum < pointer && current_individual < population.size()) {
            current_sum += population[current_individual].get_fitness() + offset;
            if (current_sum < pointer) {
                current_individual++;
            }
        }
        
        selected.push_back(current_individual);
    }
    
    return selected;
}

// Boltzmann selection (simulated annealing style)
std::vector<uint32_t> boltzmann_selection(const GenomePopulation& population,
                                        uint32_t num_parents,
                                        float temperature,
                                        std::mt19937& rng) {
    std::vector<uint32_t> selected;
    selected.reserve(num_parents);
    
    // Calculate Boltzmann probabilities
    std::vector<float> probabilities;
    probabilities.reserve(population.size());
    
    float max_fitness = population.get_best_fitness();
    float sum_exp = 0.0f;
    
    for (uint32_t i = 0; i < population.size(); ++i) {
        float exp_val = std::exp((population[i].get_fitness() - max_fitness) / temperature);
        probabilities.push_back(exp_val);
        sum_exp += exp_val;
    }
    
    // Normalize probabilities
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }
    
    // Select based on probabilities
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    
    for (uint32_t p = 0; p < num_parents; ++p) {
        float rand_val = prob_dist(rng);
        float cumulative = 0.0f;
        
        for (uint32_t i = 0; i < population.size(); ++i) {
            cumulative += probabilities[i];
            if (cumulative >= rand_val) {
                selected.push_back(i);
                break;
            }
        }
    }
    
    return selected;
}

// Diversity-based selection
std::vector<uint32_t> diversity_selection(const GenomePopulation& population,
                                        uint32_t num_parents,
                                        float diversity_weight,
                                        std::mt19937& rng) {
    std::vector<uint32_t> selected;
    selected.reserve(num_parents);
    
    // Calculate diversity scores
    std::vector<float> diversity_scores(population.size(), 0.0f);
    
    for (uint32_t i = 0; i < population.size(); ++i) {
        float min_distance = std::numeric_limits<float>::max();
        
        for (uint32_t j = 0; j < population.size(); ++j) {
            if (i != j) {
                float similarity = calculate_genome_similarity(population[i], population[j]);
                float distance = 1.0f - similarity;
                min_distance = std::min(min_distance, distance);
            }
        }
        
        diversity_scores[i] = min_distance;
    }
    
    // Combine fitness and diversity
    std::vector<float> combined_scores;
    combined_scores.reserve(population.size());
    
    for (uint32_t i = 0; i < population.size(); ++i) {
        float fitness_score = population[i].get_fitness();
        float diversity_score = diversity_scores[i];
        
        combined_scores.push_back(fitness_score + diversity_weight * diversity_score);
    }
    
    // Select based on combined scores
    float total_score = 0.0f;
    for (float score : combined_scores) {
        total_score += score;
    }
    
    if (total_score <= 0.0f) {
        // Fallback to random selection
        std::uniform_int_distribution<uint32_t> pop_dist(0, population.size() - 1);
        for (uint32_t p = 0; p < num_parents; ++p) {
            selected.push_back(pop_dist(rng));
        }
        return selected;
    }
    
    std::uniform_real_distribution<float> score_dist(0.0f, total_score);
    
    for (uint32_t p = 0; p < num_parents; ++p) {
        float score_pos = score_dist(rng);
        float current_sum = 0.0f;
        
        for (uint32_t i = 0; i < population.size(); ++i) {
            current_sum += combined_scores[i];
            if (current_sum >= score_pos) {
                selected.push_back(i);
                break;
            }
        }
    }
    
    return selected;
}

} // namespace circuit 