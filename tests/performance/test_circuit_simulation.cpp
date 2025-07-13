#include <gtest/gtest.h>
#include <chrono>
#include "circuit/core/circuit.h"
#include "circuit/core/types.h"

using namespace circuit;

class CircuitSimulationPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid = GridDimensions(32, 32);
    }
    
    GridDimensions grid;
};

// Test basic circuit simulation performance
TEST_F(CircuitSimulationPerformanceTest, BasicSimulationPerformance) {
    Circuit circuit(grid, 4, 2);
    
    // Create a moderately complex circuit
    for (int i = 0; i < 100; i++) {
        circuit.add_gate(GateType::AND, Position(i % grid.width, i / grid.width));
    }
    
    // Measure simulation time
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1000; i++) {
        std::vector<LogicState> inputs = {LogicState::HIGH, LogicState::LOW, LogicState::HIGH, LogicState::LOW};
        std::vector<LogicState> outputs;
        circuit.simulate(inputs, outputs);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "1000 simulations took " << duration.count() << " ms" << std::endl;
    
    // Should be reasonably fast
    EXPECT_LT(duration.count(), 5000);  // Less than 5 seconds
} 