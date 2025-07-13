#include "circuit/gpu/gpu_simulator.h"
#include "circuit/core/types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

namespace circuit {

// Device constants for gate evaluation
__constant__ float d_gate_delays[10] = {
    0.0f,  // NONE
    0.0f,  // INPUT  
    0.1f,  // OUTPUT
    1.0f,  // AND
    1.0f,  // OR
    0.5f,  // NOT
    1.2f,  // XOR
    1.0f,  // NAND
    1.0f,  // NOR
    1.2f   // XNOR
};

__constant__ uint8_t d_gate_input_counts[10] = {
    0, 0, 1, 2, 2, 1, 2, 2, 2, 2
};

// Device function to evaluate a single gate
__device__ SignalValue evaluate_gate_device(GateType type, SignalValue* inputs, uint8_t num_inputs) {
    switch (type) {
        case GateType::INPUT:
            return inputs[0];
        case GateType::OUTPUT:
            return inputs[0];
        case GateType::NOT:
            return (inputs[0] == 0) ? 1 : 0;
        case GateType::BUFFER:
            return inputs[0];
        case GateType::AND:
            return (num_inputs >= 2 && inputs[0] && inputs[1]) ? 1 : 0;
        case GateType::OR:
            return (num_inputs >= 2 && (inputs[0] || inputs[1])) ? 1 : 0;
        case GateType::XOR:
            return (num_inputs >= 2 && (inputs[0] ^ inputs[1])) ? 1 : 0;
        case GateType::NAND:
            return (num_inputs >= 2 && inputs[0] && inputs[1]) ? 0 : 1;
        case GateType::NOR:
            return (num_inputs >= 2 && (inputs[0] || inputs[1])) ? 0 : 1;
        case GateType::XNOR:
            return (num_inputs >= 2 && (inputs[0] ^ inputs[1])) ? 0 : 1;
        default:
            return 0;
    }
}

// Single circuit simulation kernel
__global__ void simulate_single_circuit_kernel(
    GPUCircuit* circuit,
    SignalValue* input_values,
    SignalValue* output_values,
    uint32_t* gate_states,
    float* gate_arrival_times,
    uint32_t max_iterations,
    float convergence_threshold,
    PerformanceMetrics* result
) {
    uint32_t gate_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gate_id >= circuit->num_gates) {
        return;
    }
    
    __shared__ bool convergence_reached;
    __shared__ uint32_t iteration_count;
    __shared__ float max_delay;
    __shared__ uint32_t switch_count;
    
    // Initialize shared memory
    if (threadIdx.x == 0) {
        convergence_reached = false;
        iteration_count = 0;
        max_delay = 0.0f;
        switch_count = 0;
    }
    __syncthreads();
    
    SignalValue prev_state = gate_states[gate_id];
    SignalValue new_state = prev_state;
    float arrival_time = 0.0f;
    
    // Set initial input values
    for (uint32_t i = 0; i < circuit->num_inputs; ++i) {
        if (gate_id == circuit->input_gate_indices[i]) {
            gate_states[gate_id] = input_values[i];
            gate_arrival_times[gate_id] = 0.0f;
        }
    }
    
    // Iterative simulation loop
    for (uint32_t iter = 0; iter < max_iterations && !convergence_reached; ++iter) {
        __syncthreads();
        
        // Collect inputs for this gate
        SignalValue gate_inputs[4] = {0, 0, 0, 0};
        uint8_t input_count = 0;
        float max_input_arrival = 0.0f;
        
        // Find connections to this gate
        for (uint32_t conn = 0; conn < circuit->num_connections; ++conn) {
            if (circuit->connection_to_gates[conn] == gate_id) {
                uint32_t from_gate = circuit->connection_from_gates[conn];
                if (input_count < 4) {
                    gate_inputs[input_count] = gate_states[from_gate];
                    max_input_arrival = fmaxf(max_input_arrival, gate_arrival_times[from_gate]);
                    input_count++;
                }
            }
        }
        
        // Evaluate gate if we have the required inputs
        uint8_t required_inputs = d_gate_input_counts[static_cast<int>(circuit->gate_types[gate_id])];
        if (input_count >= required_inputs || circuit->gate_types[gate_id] == GateType::INPUT) {
            new_state = evaluate_gate_device(circuit->gate_types[gate_id], gate_inputs, input_count);
            arrival_time = max_input_arrival + d_gate_delays[static_cast<int>(circuit->gate_types[gate_id])];
            
            // Update state if changed
            if (new_state != gate_states[gate_id]) {
                gate_states[gate_id] = new_state;
                gate_arrival_times[gate_id] = arrival_time;
                atomicAdd(&switch_count, 1);
            }
        }
        
        // Update maximum delay
        atomicMax(reinterpret_cast<int*>(&max_delay), __float_as_int(arrival_time));
        
        __syncthreads();
        
        // Check convergence (simplified)
        if (threadIdx.x == 0) {
            iteration_count = iter + 1;
            if (iter > 10 && switch_count < circuit->num_gates * 0.01f) {
                convergence_reached = true;
            }
        }
        
        __syncthreads();
    }
    
    // Extract output values
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < circuit->num_outputs; ++i) {
            uint32_t output_gate = circuit->output_gate_indices[i];
            output_values[i] = gate_states[output_gate];
        }
        
        // Store performance metrics
        if (result) {
            result->total_delay = max_delay;
            result->power_consumption = switch_count * 0.1f; // Simple power model
            result->gate_count = circuit->num_gates;
            result->switching_activity = switch_count;
            result->correctness_score = 1.0f; // Will be calculated by comparison
            result->correctness = 1.0f;
            result->propagation_delay = max_delay;
            result->area_cost = circuit->num_gates * 1.0f; // Simple area model
        }
    }
}

// Batch circuit simulation kernel
__global__ void simulate_circuit_batch_kernel(
    GPUCircuit* circuits,
    TestCase* test_cases,
    PerformanceMetrics* results,
    uint32_t num_circuits,
    uint32_t num_test_cases,
    GPUSimulationParams params
) {
    uint32_t circuit_idx = blockIdx.x;
    uint32_t thread_idx = threadIdx.x;
    
    if (circuit_idx >= num_circuits) {
        return;
    }
    
    GPUCircuit* circuit = &circuits[circuit_idx];
    
    // Allocate shared memory for gate states
    extern __shared__ uint32_t shared_memory[];
    SignalValue* gate_states = reinterpret_cast<SignalValue*>(shared_memory);
    SignalValue* gate_arrival_times = reinterpret_cast<SignalValue*>(
        shared_memory + circuit->num_gates);
    
    // Initialize gate states
    if (thread_idx < circuit->num_gates) {
        gate_states[thread_idx] = 0;
        gate_arrival_times[thread_idx] = 0.0f;
    }
    __syncthreads();
    
    float total_correctness = 0.0f;
    float total_delay = 0.0f;
    uint32_t total_switches = 0;
    
    // Run simulation for each test case
    for (uint32_t test_idx = 0; test_idx < num_test_cases; ++test_idx) {
        TestCase* test_case = &test_cases[test_idx];
        
        // Convert LogicState to SignalValue for inputs
        SignalValue inputs[32]; // Max inputs
        SignalValue outputs[32]; // Max outputs
        
        if (thread_idx == 0) {
            for (uint32_t i = 0; i < circuit->num_inputs && i < 32; ++i) {
                inputs[i] = static_cast<SignalValue>(test_case->inputs[i]);
            }
        }
        __syncthreads();
        
        // Simulate this test case
        PerformanceMetrics test_result;
        simulate_single_circuit_kernel<<<1, circuit->num_gates, 
            (circuit->num_gates * 2 * sizeof(SignalValue))>>>(
            circuit, inputs, outputs, gate_states, 
            reinterpret_cast<float*>(gate_arrival_times),
            params.max_simulation_steps, params.convergence_threshold, &test_result);
        
        cudaDeviceSynchronize();
        
        // Check correctness
        if (thread_idx == 0) {
            bool correct = true;
            for (uint32_t i = 0; i < circuit->num_outputs && i < 32; ++i) {
                if (outputs[i] != static_cast<SignalValue>(test_case->expected_outputs[i])) {
                    correct = false;
                    break;
                }
            }
            
            if (correct) {
                total_correctness += 1.0f;
            }
            
            total_delay += test_result.total_delay;
            total_switches += test_result.switching_activity;
        }
        __syncthreads();
    }
    
    // Store final results
    if (thread_idx == 0) {
        PerformanceMetrics* result = &results[circuit_idx];
        result->correctness_score = total_correctness / num_test_cases;
        result->correctness = result->correctness_score;
        result->total_delay = total_delay / num_test_cases;
        result->propagation_delay = result->total_delay;
        result->power_consumption = total_switches * 0.1f / num_test_cases;
        result->gate_count = circuit->num_gates;
        result->switching_activity = total_switches / num_test_cases;
        result->area_cost = circuit->num_gates * 1.0f;
    }
}

// Host function to launch circuit simulation
extern "C" void launch_circuit_simulation_kernel(
    GPUCircuit* circuits,
    TestCase* test_cases,
    PerformanceMetrics* results,
    uint32_t num_circuits,
    uint32_t num_test_cases,
    GPUSimulationParams params
) {
    // Calculate grid and block dimensions
    dim3 grid_size(num_circuits);
    dim3 block_size(params.threads_per_block);
    
    // Calculate shared memory size (assuming max 1024 gates per circuit)
    size_t shared_mem_size = 1024 * 2 * sizeof(SignalValue);
    
    // Launch kernel
    simulate_circuit_batch_kernel<<<grid_size, block_size, shared_mem_size>>>(
        circuits, test_cases, results, num_circuits, num_test_cases, params);
    
    // Check for launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
    }
}

// Utility kernel for converting genomes to circuits on GPU
__global__ void genome_to_circuit_kernel(
    GPUGenome* genomes,
    GPUCircuit* circuits,
    uint32_t num_genomes
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_genomes) {
        return;
    }
    
    GPUGenome* genome = &genomes[idx];
    GPUCircuit* circuit = &circuits[idx];
    
    // Initialize circuit
    circuit->num_gates = 0;
    circuit->num_connections = 0;
    circuit->num_inputs = genome->num_inputs;
    circuit->num_outputs = genome->num_outputs;
    circuit->grid_dims = genome->grid_dims;
    
    // Convert active genes to gates
    uint32_t gate_count = 0;
    for (uint32_t i = 0; i < genome->num_genes; ++i) {
        if (genome->is_active[i] && gate_count < 1024) { // Max gates limit
            circuit->gate_types[gate_count] = genome->gate_types[i];
            circuit->gate_positions[gate_count] = genome->positions[i];
            circuit->gate_delays[gate_count] = d_gate_delays[static_cast<int>(genome->gate_types[i])];
            circuit->gate_input_counts[gate_count] = d_gate_input_counts[static_cast<int>(genome->gate_types[i])];
            gate_count++;
        }
    }
    
    circuit->num_gates = gate_count;
}

extern "C" void launch_genome_to_circuit_kernel(
    GPUGenome* genomes,
    GPUCircuit* circuits,
    uint32_t num_genomes
) {
    uint32_t block_size = 256;
    uint32_t grid_size = (num_genomes + block_size - 1) / block_size;
    
    genome_to_circuit_kernel<<<grid_size, block_size>>>(genomes, circuits, num_genomes);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Genome to circuit kernel error: %s\n", cudaGetErrorString(error));
    }
}

} // namespace circuit 