#include "circuit/gpu/gpu_simulator.h"
#include "circuit/utils/utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

namespace circuit {

GPUMemoryManager::~GPUMemoryManager() {
    deallocate();
}

bool GPUMemoryManager::allocate(size_t size) {
    if (device_memory != nullptr) {
        deallocate();
    }
    
    cudaError_t error = cudaMalloc(&device_memory, size);
    if (error != cudaSuccess) {
        std::cerr << "GPU Memory allocation failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    allocated_size = size;
    used_size = 0;
    
    // Initialize memory to zero
    error = cudaMemset(device_memory, 0, size);
    if (error != cudaSuccess) {
        std::cerr << "GPU Memory initialization failed: " << cudaGetErrorString(error) << std::endl;
        deallocate();
        return false;
    }
    
    return true;
}

void GPUMemoryManager::deallocate() {
    if (device_memory != nullptr) {
        cudaFree(device_memory);
        device_memory = nullptr;
        allocated_size = 0;
        used_size = 0;
    }
}

void* GPUMemoryManager::get_memory(size_t size) {
    if (device_memory == nullptr || used_size + size > allocated_size) {
        return nullptr;
    }
    
    void* ptr = static_cast<char*>(device_memory) + used_size;
    used_size += size;
    
    // Align to 256-byte boundary for optimal performance
    size_t remainder = used_size % 256;
    if (remainder != 0) {
        used_size += (256 - remainder);
    }
    
    return ptr;
}

void GPUMemoryManager::reset() {
    used_size = 0;
}

namespace gpu_utils {

bool check_gpu_compatibility() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA-capable devices found" << std::endl;
        return false;
    }
    
    cudaDeviceProp props;
    error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device properties" << std::endl;
        return false;
    }
    
    // Check for minimum compute capability (3.5+)
    if (props.major < 3 || (props.major == 3 && props.minor < 5)) {
        std::cerr << "GPU compute capability " << props.major << "." << props.minor 
                  << " is insufficient (minimum 3.5 required)" << std::endl;
        return false;
    }
    
    return true;
}

std::vector<int> get_available_devices() {
    std::vector<int> devices;
    int device_count;
    
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        return devices;
    }
    
    for (int i = 0; i < device_count; ++i) {
        if (is_device_suitable(i)) {
            devices.push_back(i);
        }
    }
    
    return devices;
}

bool is_device_suitable(int device_id) {
    cudaDeviceProp props;
    cudaError_t error = cudaGetDeviceProperties(&props, device_id);
    
    if (error != cudaSuccess) {
        return false;
    }
    
    // Check compute capability, memory, and multiprocessors
    return (props.major >= 3 && (props.major > 3 || props.minor >= 5)) &&
           (props.totalGlobalMem >= 512 * 1024 * 1024) &&  // At least 512MB
           (props.multiProcessorCount >= 4);               // At least 4 SMs
}

size_t get_available_memory(int device_id) {
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        return 0;
    }
    
    size_t free_mem, total_mem;
    error = cudaMemGetInfo(&free_mem, &total_mem);
    if (error != cudaSuccess) {
        return 0;
    }
    
    return free_mem;
}

uint32_t calculate_optimal_block_size(uint32_t problem_size) {
    // For most modern GPUs, block sizes of 128, 256, or 512 work well
    // Choose based on problem size to maximize occupancy
    if (problem_size < 128) {
        return 64;
    } else if (problem_size < 1024) {
        return 128;
    } else if (problem_size < 8192) {
        return 256;
    } else {
        return 512;
    }
}

uint32_t calculate_optimal_grid_size(uint32_t problem_size, uint32_t block_size) {
    return (problem_size + block_size - 1) / block_size;
}

} // namespace gpu_utils

} // namespace circuit 