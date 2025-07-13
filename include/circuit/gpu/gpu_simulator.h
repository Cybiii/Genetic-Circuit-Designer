#pragma once

#include "../core/types.h"
#include "../core/circuit.h"
#include "../ga/genome.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <memory>
#include <vector>

namespace circuit {

// GPU memory management
struct GPUMemoryManager {
    void* device_memory;
    size_t allocated_size;
    size_t used_size;
    
    GPUMemoryManager() : device_memory(nullptr), allocated_size(0), used_size(0) {}
    ~GPUMemoryManager();
    
    bool allocate(size_t size);
    void deallocate();
    void* get_memory(size_t size);
    void reset();
};

// GPU-friendly circuit representation
struct GPUCircuit {
    // Gate data (Structure of Arrays for coalesced access)
    GateType* gate_types;
    uint32_t* gate_positions;
    float* gate_delays;
    uint8_t* gate_input_counts;
    uint8_t* gate_output_counts;
    
    // Connection data
    uint32_t* connection_from_gates;
    uint32_t* connection_to_gates;
    uint8_t* connection_from_pins;
    uint8_t* connection_to_pins;
    
    // Input/Output mapping
    uint32_t* input_gate_indices;
    uint32_t* output_gate_indices;
    
    // Circuit properties
    uint32_t num_gates;
    uint32_t num_connections;
    uint32_t num_inputs;
    uint32_t num_outputs;
    GridDimensions grid_dims;
    
    GPUCircuit() : gate_types(nullptr), gate_positions(nullptr), gate_delays(nullptr),
                  gate_input_counts(nullptr), gate_output_counts(nullptr),
                  connection_from_gates(nullptr), connection_to_gates(nullptr),
                  connection_from_pins(nullptr), connection_to_pins(nullptr),
                  input_gate_indices(nullptr), output_gate_indices(nullptr),
                  num_gates(0), num_connections(0), num_inputs(0), num_outputs(0) {}
};

// GPU-friendly genome representation
struct GPUGenome {
    Gene* genes;
    uint32_t num_genes;
    uint32_t num_inputs;
    uint32_t num_outputs;
    GridDimensions grid_dims;
    float fitness;
    
    GPUGenome() : genes(nullptr), num_genes(0), num_inputs(0), num_outputs(0), fitness(0.0f) {}
};

// GPU simulation parameters
struct GPUSimulationParams {
    uint32_t max_simulation_steps;
    uint32_t batch_size;
    uint32_t threads_per_block;
    uint32_t blocks_per_grid;
    float convergence_threshold;
    
    GPUSimulationParams() : max_simulation_steps(1000), batch_size(1024),
                           threads_per_block(256), blocks_per_grid(64),
                           convergence_threshold(1e-6f) {}
};

// GPU genetic algorithm parameters
struct GPUGeneticParams {
    uint32_t population_size;
    uint32_t max_generations;
    float mutation_rate;
    float crossover_rate;
    uint32_t tournament_size;
    uint32_t elite_count;
    
    GPUGeneticParams() : population_size(1024), max_generations(1000),
                        mutation_rate(0.1f), crossover_rate(0.8f),
                        tournament_size(4), elite_count(10) {}
};

// Main GPU simulator class
class GPUSimulator {
public:
    GPUSimulator();
    ~GPUSimulator();
    
    // Initialization
    bool initialize(const GPUSimulationParams& params);
    void cleanup();
    
    // Device management
    bool select_device(int device_id = 0);
    void get_device_info();
    
    // Memory management
    bool allocate_memory(uint32_t max_circuits, uint32_t max_genes_per_circuit);
    void deallocate_memory();
    
    // Circuit simulation
    bool simulate_circuit_batch(const std::vector<Circuit>& circuits,
                               const std::vector<TestCase>& test_cases,
                               std::vector<PerformanceMetrics>& results);
    
    bool simulate_genome_batch(const std::vector<Genome>& genomes,
                              const std::vector<TestCase>& test_cases,
                              std::vector<PerformanceMetrics>& results);
    
    // Genetic algorithm operations
    bool genetic_algorithm_step(std::vector<Genome>& population,
                               const std::vector<TestCase>& test_cases,
                               const GPUGeneticParams& params);
    
    bool tournament_selection(const std::vector<Genome>& population,
                             std::vector<uint32_t>& selected_indices,
                             const GPUGeneticParams& params);
    
    bool crossover_operation(const std::vector<Genome>& parents,
                            std::vector<Genome>& offspring,
                            const GPUGeneticParams& params);
    
    bool mutation_operation(std::vector<Genome>& genomes,
                           const GPUGeneticParams& params);
    
    // Fitness evaluation
    bool evaluate_fitness_batch(std::vector<Genome>& genomes,
                               const std::vector<TestCase>& test_cases,
                               const FitnessComponents& fitness_weights);
    
    // Utility functions
    bool copy_circuits_to_gpu(const std::vector<Circuit>& circuits);
    bool copy_genomes_to_gpu(const std::vector<Genome>& genomes);
    bool copy_test_cases_to_gpu(const std::vector<TestCase>& test_cases);
    
    bool copy_results_from_gpu(std::vector<PerformanceMetrics>& results);
    bool copy_genomes_from_gpu(std::vector<Genome>& genomes);
    
    // Performance monitoring
    struct GPUPerformanceMetrics {
        float kernel_execution_time;
        float memory_transfer_time;
        float total_time;
        uint32_t circuits_simulated;
        float circuits_per_second;
        
        GPUPerformanceMetrics() : kernel_execution_time(0.0f), memory_transfer_time(0.0f),
                                 total_time(0.0f), circuits_simulated(0), circuits_per_second(0.0f) {}
    };
    
    GPUPerformanceMetrics get_performance_metrics() const { return performance_metrics_; }
    void reset_performance_metrics();
    
private:
    // Device properties
    int device_id_;
    cudaDeviceProp device_props_;
    
    // Memory management
    GPUMemoryManager memory_manager_;
    
    // GPU data structures
    GPUCircuit* d_circuits_;
    GPUGenome* d_genomes_;
    TestCase* d_test_cases_;
    PerformanceMetrics* d_results_;
    
    // Simulation parameters
    GPUSimulationParams sim_params_;
    
    // Random number generation
    curandGenerator_t rng_generator_;
    
    // CUDA streams for overlapping operations
    cudaStream_t simulation_stream_;
    cudaStream_t memory_stream_;
    
    // Performance tracking
    GPUPerformanceMetrics performance_metrics_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    
    // Helper methods
    bool convert_circuit_to_gpu(const Circuit& circuit, GPUCircuit& gpu_circuit);
    bool convert_genome_to_gpu(const Genome& genome, GPUGenome& gpu_genome);
    bool convert_gpu_results(const PerformanceMetrics* gpu_results, 
                           std::vector<PerformanceMetrics>& results);
    
    // Error handling
    bool check_cuda_error(cudaError_t error, const char* operation);
    void log_cuda_error(cudaError_t error, const char* operation);
};

// CUDA kernel declarations (implemented in .cu files)
extern "C" {
    // Circuit simulation kernels
    void launch_circuit_simulation_kernel(GPUCircuit* circuits, TestCase* test_cases,
                                        PerformanceMetrics* results, uint32_t num_circuits,
                                        uint32_t num_test_cases, GPUSimulationParams params);
    
    // Genetic algorithm kernels
    void launch_tournament_selection_kernel(GPUGenome* genomes, uint32_t* selected_indices,
                                          uint32_t population_size, uint32_t tournament_size,
                                          curandState* rand_states);
    
    void launch_crossover_kernel(GPUGenome* parents, GPUGenome* offspring,
                               uint32_t num_pairs, float crossover_rate,
                               curandState* rand_states);
    
    void launch_mutation_kernel(GPUGenome* genomes, uint32_t population_size,
                              float mutation_rate, curandState* rand_states);
    
    void launch_fitness_evaluation_kernel(GPUGenome* genomes, TestCase* test_cases,
                                        PerformanceMetrics* results, uint32_t population_size,
                                        uint32_t num_test_cases, FitnessComponents weights);
    
    // Utility kernels
    void launch_genome_to_circuit_kernel(GPUGenome* genomes, GPUCircuit* circuits,
                                       uint32_t num_genomes);
    
    void launch_random_init_kernel(curandState* rand_states, uint32_t num_states,
                                 unsigned long long seed);
}

// Utility functions for GPU operations
namespace gpu_utils {
    bool check_gpu_compatibility();
    std::vector<int> get_available_devices();
    bool is_device_suitable(int device_id);
    size_t get_available_memory(int device_id);
    uint32_t calculate_optimal_block_size(uint32_t problem_size);
    uint32_t calculate_optimal_grid_size(uint32_t problem_size, uint32_t block_size);
}

} // namespace circuit 