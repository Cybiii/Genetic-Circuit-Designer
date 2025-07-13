#include "circuit/gpu/gpu_simulator.h"
#include "circuit/core/types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cstdio>

namespace circuit {

// Tournament selection kernel
__global__ void tournament_selection_kernel(
    GPUGenome* genomes,
    uint32_t* selected_indices,
    uint32_t population_size,
    uint32_t tournament_size,
    curandState* rand_states
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= population_size) {
        return;
    }
    
    curandState local_state = rand_states[idx];
    
    // Select random individuals for tournament
    uint32_t best_idx = curand(&local_state) % population_size;
    float best_fitness = genomes[best_idx].fitness;
    
    for (uint32_t i = 1; i < tournament_size; ++i) {
        uint32_t candidate_idx = curand(&local_state) % population_size;
        float candidate_fitness = genomes[candidate_idx].fitness;
        
        if (candidate_fitness > best_fitness) {
            best_idx = candidate_idx;
            best_fitness = candidate_fitness;
        }
    }
    
    selected_indices[idx] = best_idx;
    rand_states[idx] = local_state;
}

// Crossover kernel
__global__ void crossover_kernel(
    GPUGenome* parents,
    GPUGenome* offspring,
    uint32_t num_pairs,
    float crossover_rate,
    curandState* rand_states
) {
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pair_idx >= num_pairs) {
        return;
    }
    
    curandState local_state = rand_states[pair_idx];
    
    uint32_t parent1_idx = pair_idx * 2;
    uint32_t parent2_idx = pair_idx * 2 + 1;
    uint32_t offspring1_idx = pair_idx * 2;
    uint32_t offspring2_idx = pair_idx * 2 + 1;
    
    GPUGenome* parent1 = &parents[parent1_idx];
    GPUGenome* parent2 = &parents[parent2_idx];
    GPUGenome* child1 = &offspring[offspring1_idx];
    GPUGenome* child2 = &offspring[offspring2_idx];
    
    // Copy parent data to offspring
    *child1 = *parent1;
    *child2 = *parent2;
    
    // Perform crossover if random chance
    if (curand_uniform(&local_state) < crossover_rate) {
        // Single-point crossover
        uint32_t crossover_point = curand(&local_state) % parent1->num_genes;
        
        // Swap genes after crossover point
        for (uint32_t i = crossover_point; i < parent1->num_genes; ++i) {
            // Swap gate types
            GateType temp_gate = child1->gate_types[i];
            child1->gate_types[i] = child2->gate_types[i];
            child2->gate_types[i] = temp_gate;
            
            // Swap positions
            uint32_t temp_pos = child1->positions[i];
            child1->positions[i] = child2->positions[i];
            child2->positions[i] = temp_pos;
            
            // Swap active flags
            bool temp_active = child1->is_active[i];
            child1->is_active[i] = child2->is_active[i];
            child2->is_active[i] = temp_active;
        }
    }
    
    rand_states[pair_idx] = local_state;
}

// Mutation kernel
__global__ void mutation_kernel(
    GPUGenome* genomes,
    uint32_t population_size,
    float mutation_rate,
    curandState* rand_states
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= population_size) {
        return;
    }
    
    curandState local_state = rand_states[idx];
    GPUGenome* genome = &genomes[idx];
    
    // Mutate each gene with probability mutation_rate
    for (uint32_t gene_idx = 0; gene_idx < genome->num_genes; ++gene_idx) {
        if (curand_uniform(&local_state) < mutation_rate) {
            // Choose mutation type randomly
            uint32_t mutation_type = curand(&local_state) % 4;
            
            switch (mutation_type) {
                case 0: // Gate type mutation
                    if (genome->is_active[gene_idx]) {
                        genome->gate_types[gene_idx] = static_cast<GateType>(
                            (curand(&local_state) % 7) + 3); // AND to XNOR
                    }
                    break;
                    
                case 1: // Position mutation
                    {
                        uint32_t grid_size = genome->grid_dims.width * genome->grid_dims.height;
                        genome->positions[gene_idx] = curand(&local_state) % grid_size;
                    }
                    break;
                    
                case 2: // Activation mutation
                    genome->is_active[gene_idx] = !genome->is_active[gene_idx];
                    break;
                    
                case 3: // Connection mutation (simplified)
                    if (genome->is_active[gene_idx] && gene_idx > 0) {
                        // Randomly connect to previous gene
                        uint32_t connection_target = curand(&local_state) % gene_idx;
                        // Would need to modify connection arrays here
                    }
                    break;
            }
        }
    }
    
    rand_states[idx] = local_state;
}

// Fitness evaluation kernel
__global__ void fitness_evaluation_kernel(
    GPUGenome* genomes,
    TestCase* test_cases,
    PerformanceMetrics* results,
    uint32_t population_size,
    uint32_t num_test_cases,
    FitnessComponents weights
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= population_size) {
        return;
    }
    
    GPUGenome* genome = &genomes[idx];
    PerformanceMetrics* result = &results[idx];
    
    // Initialize metrics
    result->correctness_score = 0.0f;
    result->total_delay = 0.0f;
    result->power_consumption = 0.0f;
    result->gate_count = 0;
    result->switching_activity = 0;
    result->area_cost = 0.0f;
    
    // Count active gates
    for (uint32_t i = 0; i < genome->num_genes; ++i) {
        if (genome->is_active[i]) {
            result->gate_count++;
            result->area_cost += 1.0f; // Simple area model
        }
    }
    
    // Simple fitness calculation based on gate count and complexity
    float complexity_penalty = result->gate_count * 0.1f;
    float diversity_bonus = (genome->num_genes > 0) ? 
        (float)result->gate_count / genome->num_genes : 0.0f;
    
    // Calculate composite fitness
    float fitness = 0.0f;
    fitness += weights.correctness_weight * result->correctness_score;
    fitness -= weights.delay_weight * result->total_delay;
    fitness -= weights.power_weight * result->power_consumption;
    fitness -= weights.area_weight * complexity_penalty;
    fitness += 0.1f * diversity_bonus; // Small diversity bonus
    
    genome->fitness = fmaxf(0.0f, fitness); // Ensure non-negative
    
    // Copy aliased fields
    result->correctness = result->correctness_score;
    result->propagation_delay = result->total_delay;
}

// Random state initialization kernel
__global__ void random_init_kernel(
    curandState* rand_states,
    uint32_t num_states,
    unsigned long long seed
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_states) {
        return;
    }
    
    // Initialize random state for this thread
    curand_init(seed, idx, 0, &rand_states[idx]);
}

// Population initialization kernel
__global__ void population_init_kernel(
    GPUGenome* genomes,
    uint32_t population_size,
    GridDimensions grid_dims,
    uint32_t num_inputs,
    uint32_t num_outputs,
    curandState* rand_states
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= population_size) {
        return;
    }
    
    curandState local_state = rand_states[idx];
    GPUGenome* genome = &genomes[idx];
    
    // Initialize genome properties
    genome->num_genes = grid_dims.width * grid_dims.height;
    genome->num_inputs = num_inputs;
    genome->num_outputs = num_outputs;
    genome->grid_dims = grid_dims;
    genome->fitness = 0.0f;
    
    // Initialize genes randomly
    for (uint32_t i = 0; i < genome->num_genes; ++i) {
        // Random gate type
        genome->gate_types[i] = static_cast<GateType>(
            (curand(&local_state) % 7) + 3); // AND to XNOR
        
        // Position based on grid layout
        uint32_t x = i % grid_dims.width;
        uint32_t y = i / grid_dims.width;
        genome->positions[i] = y * grid_dims.width + x;
        
        // Random activation (sparse)
        genome->is_active[i] = (curand_uniform(&local_state) < 0.3f);
        
        // Initialize connections (simplified)
        // Would need proper connection initialization here
    }
    
    // Ensure input and output gates are active
    for (uint32_t i = 0; i < num_inputs && i < genome->num_genes; ++i) {
        genome->gate_types[i] = GateType::INPUT;
        genome->is_active[i] = true;
    }
    
    for (uint32_t i = 0; i < num_outputs && i + num_inputs < genome->num_genes; ++i) {
        genome->gate_types[i + num_inputs] = GateType::OUTPUT;
        genome->is_active[i + num_inputs] = true;
    }
    
    rand_states[idx] = local_state;
}

// Host function implementations
extern "C" void launch_tournament_selection_kernel(
    GPUGenome* genomes,
    uint32_t* selected_indices,
    uint32_t population_size,
    uint32_t tournament_size,
    curandState* rand_states
) {
    uint32_t block_size = 256;
    uint32_t grid_size = (population_size + block_size - 1) / block_size;
    
    tournament_selection_kernel<<<grid_size, block_size>>>(
        genomes, selected_indices, population_size, tournament_size, rand_states);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Tournament selection kernel error: %s\n", cudaGetErrorString(error));
    }
}

extern "C" void launch_crossover_kernel(
    GPUGenome* parents,
    GPUGenome* offspring,
    uint32_t num_pairs,
    float crossover_rate,
    curandState* rand_states
) {
    uint32_t block_size = 256;
    uint32_t grid_size = (num_pairs + block_size - 1) / block_size;
    
    crossover_kernel<<<grid_size, block_size>>>(
        parents, offspring, num_pairs, crossover_rate, rand_states);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Crossover kernel error: %s\n", cudaGetErrorString(error));
    }
}

extern "C" void launch_mutation_kernel(
    GPUGenome* genomes,
    uint32_t population_size,
    float mutation_rate,
    curandState* rand_states
) {
    uint32_t block_size = 256;
    uint32_t grid_size = (population_size + block_size - 1) / block_size;
    
    mutation_kernel<<<grid_size, block_size>>>(
        genomes, population_size, mutation_rate, rand_states);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Mutation kernel error: %s\n", cudaGetErrorString(error));
    }
}

extern "C" void launch_fitness_evaluation_kernel(
    GPUGenome* genomes,
    TestCase* test_cases,
    PerformanceMetrics* results,
    uint32_t population_size,
    uint32_t num_test_cases,
    FitnessComponents weights
) {
    uint32_t block_size = 256;
    uint32_t grid_size = (population_size + block_size - 1) / block_size;
    
    fitness_evaluation_kernel<<<grid_size, block_size>>>(
        genomes, test_cases, results, population_size, num_test_cases, weights);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Fitness evaluation kernel error: %s\n", cudaGetErrorString(error));
    }
}

extern "C" void launch_random_init_kernel(
    curandState* rand_states,
    uint32_t num_states,
    unsigned long long seed
) {
    uint32_t block_size = 256;
    uint32_t grid_size = (num_states + block_size - 1) / block_size;
    
    random_init_kernel<<<grid_size, block_size>>>(rand_states, num_states, seed);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Random init kernel error: %s\n", cudaGetErrorString(error));
    }
}

} // namespace circuit 