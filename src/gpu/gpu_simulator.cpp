#include "circuit/gpu/gpu_simulator.h"
#include "circuit/utils/utils.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <cassert>
#include <chrono>

namespace circuit {

GPUSimulator::GPUSimulator() 
    : device_id_(-1), d_circuits_(nullptr), d_genomes_(nullptr),
      d_test_cases_(nullptr), d_results_(nullptr), rng_generator_(nullptr),
      simulation_stream_(nullptr), memory_stream_(nullptr) {
    
    // Initialize CUDA events for timing
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

GPUSimulator::~GPUSimulator() {
    cleanup();
    
    // Destroy CUDA events
    if (start_event_) cudaEventDestroy(start_event_);
    if (stop_event_) cudaEventDestroy(stop_event_);
}

bool GPUSimulator::initialize(const GPUSimulationParams& params) {
    if (!gpu_utils::check_gpu_compatibility()) {
        std::cerr << "GPU not compatible for circuit simulation" << std::endl;
        return false;
    }
    
    sim_params_ = params;
    
    // Select and initialize device
    if (!select_device()) {
        return false;
    }
    
    // Create CUDA streams
    cudaError_t error = cudaStreamCreate(&simulation_stream_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create simulation stream: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaStreamCreate(&memory_stream_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create memory stream: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Initialize random number generator
    curandStatus_t curand_status = curandCreateGenerator(&rng_generator_, CURAND_RNG_PSEUDO_DEFAULT);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuRAND generator" << std::endl;
        return false;
    }
    
    curand_status = curandSetPseudoRandomGeneratorSeed(rng_generator_, 12345ULL);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        std::cerr << "Failed to set cuRAND seed" << std::endl;
        return false;
    }
    
    return true;
}

void GPUSimulator::cleanup() {
    // Clean up device memory
    memory_manager_.deallocate();
    
    // Destroy cuRAND generator
    if (rng_generator_) {
        curandDestroyGenerator(rng_generator_);
        rng_generator_ = nullptr;
    }
    
    // Destroy CUDA streams
    if (simulation_stream_) {
        cudaStreamDestroy(simulation_stream_);
        simulation_stream_ = nullptr;
    }
    
    if (memory_stream_) {
        cudaStreamDestroy(memory_stream_);
        memory_stream_ = nullptr;
    }
    
    // Reset device pointers
    d_circuits_ = nullptr;
    d_genomes_ = nullptr;
    d_test_cases_ = nullptr;
    d_results_ = nullptr;
}

bool GPUSimulator::select_device(int device_id) {
    std::vector<int> available_devices = gpu_utils::get_available_devices();
    
    if (available_devices.empty()) {
        std::cerr << "No suitable CUDA devices found" << std::endl;
        return false;
    }
    
    // Use specified device or best available
    int target_device = (device_id >= 0) ? device_id : available_devices[0];
    
    cudaError_t error = cudaSetDevice(target_device);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device " << target_device << ": " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Get device properties
    error = cudaGetDeviceProperties(&device_props_, target_device);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    device_id_ = target_device;
    
    // Log device information
    std::cout << "Using GPU Device " << device_id_ << ": " << device_props_.name << std::endl;
    std::cout << "  Compute Capability: " << device_props_.major << "." << device_props_.minor << std::endl;
    std::cout << "  Total Global Memory: " << device_props_.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << device_props_.multiProcessorCount << std::endl;
    std::cout << "  Max Threads per Block: " << device_props_.maxThreadsPerBlock << std::endl;
    
    return true;
}

bool GPUSimulator::allocate_memory(uint32_t max_circuits, uint32_t max_genes_per_circuit) {
    // Calculate memory requirements
    size_t circuit_mem = max_circuits * sizeof(GPUCircuit);
    size_t genome_mem = max_circuits * sizeof(GPUGenome);
    size_t test_cases_mem = 1000 * sizeof(TestCase); // Max 1000 test cases
    size_t results_mem = max_circuits * sizeof(PerformanceMetrics);
    
    // Additional memory for circuit data arrays
    size_t gate_data_mem = max_circuits * max_genes_per_circuit * sizeof(GateType) * 8; // Multiple arrays
    
    size_t total_memory = circuit_mem + genome_mem + test_cases_mem + results_mem + gate_data_mem;
    
    if (!memory_manager_.allocate(total_memory)) {
        std::cerr << "Failed to allocate " << total_memory / (1024*1024) << " MB of GPU memory" << std::endl;
        return false;
    }
    
    // Assign memory regions
    d_circuits_ = static_cast<GPUCircuit*>(memory_manager_.get_memory(circuit_mem));
    d_genomes_ = static_cast<GPUGenome*>(memory_manager_.get_memory(genome_mem));
    d_test_cases_ = static_cast<TestCase*>(memory_manager_.get_memory(test_cases_mem));
    d_results_ = static_cast<PerformanceMetrics*>(memory_manager_.get_memory(results_mem));
    
    if (!d_circuits_ || !d_genomes_ || !d_test_cases_ || !d_results_) {
        std::cerr << "Failed to allocate GPU memory regions" << std::endl;
        return false;
    }
    
    std::cout << "Allocated " << total_memory / (1024*1024) << " MB GPU memory for " 
              << max_circuits << " circuits" << std::endl;
    
    return true;
}

bool GPUSimulator::simulate_circuit_batch(
    const std::vector<Circuit>& circuits,
    const std::vector<TestCase>& test_cases,
    std::vector<PerformanceMetrics>& results
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (circuits.empty() || test_cases.empty()) {
        return false;
    }
    
    uint32_t num_circuits = circuits.size();
    uint32_t num_test_cases = test_cases.size();
    
    // Start timing
    cudaEventRecord(start_event_, simulation_stream_);
    
    // Convert circuits to GPU format
    std::vector<GPUCircuit> gpu_circuits(num_circuits);
    for (uint32_t i = 0; i < num_circuits; ++i) {
        if (!convert_circuit_to_gpu(circuits[i], gpu_circuits[i])) {
            std::cerr << "Failed to convert circuit " << i << " to GPU format" << std::endl;
            return false;
        }
    }
    
    // Copy data to device
    cudaError_t error = cudaMemcpyAsync(d_circuits_, gpu_circuits.data(), 
                                      num_circuits * sizeof(GPUCircuit),
                                      cudaMemcpyHostToDevice, memory_stream_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy circuits to device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    error = cudaMemcpyAsync(d_test_cases_, test_cases.data(),
                          num_test_cases * sizeof(TestCase),
                          cudaMemcpyHostToDevice, memory_stream_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy test cases to device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Synchronize memory transfers
    cudaStreamSynchronize(memory_stream_);
    
    // Launch simulation kernel
    launch_circuit_simulation_kernel(d_circuits_, d_test_cases_, d_results_,
                                   num_circuits, num_test_cases, sim_params_);
    
    // Wait for kernel completion
    cudaStreamSynchronize(simulation_stream_);
    
    // Copy results back to host
    results.resize(num_circuits);
    error = cudaMemcpyAsync(results.data(), d_results_,
                          num_circuits * sizeof(PerformanceMetrics),
                          cudaMemcpyDeviceToHost, memory_stream_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy results from device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    cudaStreamSynchronize(memory_stream_);
    
    // Stop timing
    cudaEventRecord(stop_event_, simulation_stream_);
    cudaEventSynchronize(stop_event_);
    
    // Calculate performance metrics
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event_, stop_event_);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    performance_metrics_.kernel_execution_time = kernel_time_ms;
    performance_metrics_.memory_transfer_time = total_time_ms - kernel_time_ms;
    performance_metrics_.total_time = total_time_ms;
    performance_metrics_.circuits_simulated = num_circuits;
    performance_metrics_.circuits_per_second = (num_circuits * 1000.0f) / total_time_ms;
    
    return true;
}

bool GPUSimulator::simulate_genome_batch(
    const std::vector<Genome>& genomes,
    const std::vector<TestCase>& test_cases,
    std::vector<PerformanceMetrics>& results
) {
    if (genomes.empty() || test_cases.empty()) {
        return false;
    }
    
    uint32_t num_genomes = genomes.size();
    uint32_t num_test_cases = test_cases.size();
    
    // Convert genomes to GPU format
    std::vector<GPUGenome> gpu_genomes(num_genomes);
    for (uint32_t i = 0; i < num_genomes; ++i) {
        if (!convert_genome_to_gpu(genomes[i], gpu_genomes[i])) {
            std::cerr << "Failed to convert genome " << i << " to GPU format" << std::endl;
            return false;
        }
    }
    
    // Copy genomes to device
    cudaError_t error = cudaMemcpyAsync(d_genomes_, gpu_genomes.data(),
                                      num_genomes * sizeof(GPUGenome),
                                      cudaMemcpyHostToDevice, memory_stream_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy genomes to device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Convert genomes to circuits on GPU
    launch_genome_to_circuit_kernel(d_genomes_, d_circuits_, num_genomes);
    
    // Copy test cases to device
    error = cudaMemcpyAsync(d_test_cases_, test_cases.data(),
                          num_test_cases * sizeof(TestCase),
                          cudaMemcpyHostToDevice, memory_stream_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy test cases to device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    cudaStreamSynchronize(memory_stream_);
    
    // Launch simulation kernel
    launch_circuit_simulation_kernel(d_circuits_, d_test_cases_, d_results_,
                                   num_genomes, num_test_cases, sim_params_);
    
    cudaStreamSynchronize(simulation_stream_);
    
    // Copy results back
    results.resize(num_genomes);
    error = cudaMemcpyAsync(results.data(), d_results_,
                          num_genomes * sizeof(PerformanceMetrics),
                          cudaMemcpyDeviceToHost, memory_stream_);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy results from device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    cudaStreamSynchronize(memory_stream_);
    
    return true;
}

bool GPUSimulator::convert_circuit_to_gpu(const Circuit& circuit, GPUCircuit& gpu_circuit) {
    // This is a simplified conversion - in a real implementation,
    // you'd need to properly extract gate and connection data from the Circuit class
    
    gpu_circuit.num_gates = circuit.get_gate_count();
    gpu_circuit.num_connections = circuit.get_connection_count();
    gpu_circuit.num_inputs = circuit.get_input_count();
    gpu_circuit.num_outputs = circuit.get_output_count();
    gpu_circuit.grid_dims = circuit.get_grid_dimensions();
    
    // For now, create dummy data - real implementation would extract from circuit
    gpu_circuit.gate_types = nullptr; // Would point to device memory
    gpu_circuit.gate_positions = nullptr;
    gpu_circuit.gate_delays = nullptr;
    gpu_circuit.gate_input_counts = nullptr;
    gpu_circuit.gate_output_counts = nullptr;
    gpu_circuit.connection_from_gates = nullptr;
    gpu_circuit.connection_to_gates = nullptr;
    gpu_circuit.connection_from_pins = nullptr;
    gpu_circuit.connection_to_pins = nullptr;
    gpu_circuit.input_gate_indices = nullptr;
    gpu_circuit.output_gate_indices = nullptr;
    
    return true;
}

bool GPUSimulator::convert_genome_to_gpu(const Genome& genome, GPUGenome& gpu_genome) {
    // Convert genome data to GPU-friendly format
    gpu_genome.num_genes = genome.get_genes().size();
    gpu_genome.num_inputs = genome.get_num_inputs();
    gpu_genome.num_outputs = genome.get_num_outputs();
    gpu_genome.grid_dims = genome.get_grid_dimensions();
    gpu_genome.fitness = genome.get_fitness();
    
    // For actual implementation, would need to copy gene data arrays
    gpu_genome.gate_types = nullptr; // Would point to device memory
    gpu_genome.positions = nullptr;
    gpu_genome.input_connections = nullptr;
    gpu_genome.output_connections = nullptr;
    gpu_genome.is_active = nullptr;
    
    return true;
}

void GPUSimulator::reset_performance_metrics() {
    performance_metrics_ = GPUPerformanceMetrics();
}

DeviceInfo GPUSimulator::get_device_info() const {
    DeviceInfo info;
    if (device_id_ >= 0) {
        info.device_id = device_id_;
        info.name = device_props_.name;
        info.compute_capability_major = device_props_.major;
        info.compute_capability_minor = device_props_.minor;
        info.total_global_memory = device_props_.totalGlobalMem;
        info.multiprocessor_count = device_props_.multiProcessorCount;
        info.max_threads_per_block = device_props_.maxThreadsPerBlock;
        info.max_threads_per_multiprocessor = device_props_.maxThreadsPerMultiProcessor;
        info.memory_clock_rate = device_props_.memoryClockRate;
        info.memory_bus_width = device_props_.memoryBusWidth;
    }
    return info;
}

bool GPUSimulator::check_cuda_error(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        log_cuda_error(error, operation);
        return false;
    }
    return true;
}

void GPUSimulator::log_cuda_error(cudaError_t error, const char* operation) {
    std::cerr << "CUDA Error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
}

} // namespace circuit 