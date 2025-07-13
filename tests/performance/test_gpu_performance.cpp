#include <gtest/gtest.h>
#include <chrono>
#include "circuit/gpu/gpu_simulator.h"
#include "circuit/core/types.h"

using namespace circuit;

class GPUPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!GPUSimulator::is_cuda_available()) {
            GTEST_SKIP() << "CUDA not available, skipping GPU performance tests";
        }
        
        simulator = std::make_unique<GPUSimulator>();
        ASSERT_TRUE(simulator->initialize());
        
        grid = GridDimensions(16, 16);
    }
    
    void TearDown() override {
        simulator.reset();
    }
    
    std::unique_ptr<GPUSimulator> simulator;
    GridDimensions grid;
};

// Test GPU memory operations performance
TEST_F(GPUPerformanceTest, MemoryOperationsPerformance) {
    const size_t data_size = 1024 * 1024;  // 1MB
    std::vector<float> host_data(data_size / sizeof(float), 1.0f);
    
    // Measure memory allocation
    auto start = std::chrono::high_resolution_clock::now();
    
    void* device_ptr = simulator->allocate_device_memory(data_size);
    ASSERT_NE(device_ptr, nullptr);
    
    auto alloc_end = std::chrono::high_resolution_clock::now();
    
    // Measure host to device transfer
    EXPECT_TRUE(simulator->copy_to_device(device_ptr, host_data.data(), data_size));
    
    auto h2d_end = std::chrono::high_resolution_clock::now();
    
    // Measure device to host transfer
    std::vector<float> result_data(host_data.size());
    EXPECT_TRUE(simulator->copy_from_device(result_data.data(), device_ptr, data_size));
    
    auto d2h_end = std::chrono::high_resolution_clock::now();
    
    // Cleanup
    simulator->free_device_memory(device_ptr);
    
    auto cleanup_end = std::chrono::high_resolution_clock::now();
    
    // Calculate times
    auto alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - start);
    auto h2d_time = std::chrono::duration_cast<std::chrono::microseconds>(h2d_end - alloc_end);
    auto d2h_time = std::chrono::duration_cast<std::chrono::microseconds>(d2h_end - h2d_end);
    auto cleanup_time = std::chrono::duration_cast<std::chrono::microseconds>(cleanup_end - d2h_end);
    
    std::cout << "GPU Memory Operations Performance:" << std::endl;
    std::cout << "  Allocation: " << alloc_time.count() << " μs" << std::endl;
    std::cout << "  Host->Device: " << h2d_time.count() << " μs" << std::endl;
    std::cout << "  Device->Host: " << d2h_time.count() << " μs" << std::endl;
    std::cout << "  Cleanup: " << cleanup_time.count() << " μs" << std::endl;
    
    // Verify data integrity
    EXPECT_EQ(result_data, host_data);
    
    // Performance expectations (adjust based on hardware)
    EXPECT_LT(alloc_time.count(), 1000);    // < 1ms
    EXPECT_LT(h2d_time.count(), 10000);     // < 10ms
    EXPECT_LT(d2h_time.count(), 10000);     // < 10ms
    EXPECT_LT(cleanup_time.count(), 1000);  // < 1ms
} 