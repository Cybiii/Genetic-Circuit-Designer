#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 1;
    }
    
    std::cout << "=== RTX 3060 GPU Specifications ===" << std::endl;
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    std::cout << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        
        if (error != cudaSuccess) {
            std::cout << "Error getting device properties: " << cudaGetErrorString(error) << std::endl;
            continue;
        }
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  CUDA Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  CUDA Cores (est.): " << prop.multiProcessorCount * 128 << std::endl; // Approximate for RTX 3060
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Block Dimensions: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
        std::cout << "  Max Grid Dimensions: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
        std::cout << "  Clock Rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Memory Bandwidth: " << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6) << " GB/s" << std::endl;
        std::cout << "  L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  Compute Mode: " << prop.computeMode << std::endl;
        std::cout << "  Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "  ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
        std::cout << "  Unified Addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
    }
    
    return 0;
} 