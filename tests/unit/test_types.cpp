#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "circuit/core/types.h"

using namespace circuit;

class TypesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for each test
    }
    
    void TearDown() override {
        // Cleanup after each test
    }
};

// Test GateType enum
TEST_F(TypesTest, GateTypeValues) {
    EXPECT_EQ(static_cast<int>(GateType::AND), 0);
    EXPECT_EQ(static_cast<int>(GateType::OR), 1);
    EXPECT_EQ(static_cast<int>(GateType::NOT), 2);
    EXPECT_EQ(static_cast<int>(GateType::XOR), 3);
    EXPECT_EQ(static_cast<int>(GateType::NAND), 4);
    EXPECT_EQ(static_cast<int>(GateType::NOR), 5);
    EXPECT_EQ(static_cast<int>(GateType::XNOR), 6);
    EXPECT_EQ(static_cast<int>(GateType::BUFFER), 7);
    EXPECT_EQ(static_cast<int>(GateType::INPUT), 8);
    EXPECT_EQ(static_cast<int>(GateType::OUTPUT), 9);
}

// Test Position struct
TEST_F(TypesTest, PositionConstruction) {
    Position pos1;
    EXPECT_EQ(pos1.x, 0);
    EXPECT_EQ(pos1.y, 0);
    
    Position pos2(5, 10);
    EXPECT_EQ(pos2.x, 5);
    EXPECT_EQ(pos2.y, 10);
}

TEST_F(TypesTest, PositionEquality) {
    Position pos1(3, 7);
    Position pos2(3, 7);
    Position pos3(4, 7);
    
    EXPECT_EQ(pos1, pos2);
    EXPECT_NE(pos1, pos3);
}

// Test GridDimensions struct
TEST_F(TypesTest, GridDimensionsConstruction) {
    GridDimensions grid1;
    EXPECT_EQ(grid1.width, 0);
    EXPECT_EQ(grid1.height, 0);
    
    GridDimensions grid2(16, 32);
    EXPECT_EQ(grid2.width, 16);
    EXPECT_EQ(grid2.height, 32);
}

TEST_F(TypesTest, GridDimensionsValidation) {
    GridDimensions grid(10, 20);
    
    EXPECT_TRUE(grid.is_valid_position(Position(0, 0)));
    EXPECT_TRUE(grid.is_valid_position(Position(9, 19)));
    EXPECT_FALSE(grid.is_valid_position(Position(10, 19)));
    EXPECT_FALSE(grid.is_valid_position(Position(9, 20)));
    EXPECT_FALSE(grid.is_valid_position(Position(-1, 0)));
    EXPECT_FALSE(grid.is_valid_position(Position(0, -1)));
}

// Test Gate struct
TEST_F(TypesTest, GateConstruction) {
    Gate gate1;
    EXPECT_EQ(gate1.id, 0);
    EXPECT_EQ(gate1.type, GateType::AND);
    EXPECT_EQ(gate1.position.x, 0);
    EXPECT_EQ(gate1.position.y, 0);
    EXPECT_EQ(gate1.delay, 0.0f);
    EXPECT_EQ(gate1.output_state, LogicState::LOW);
    EXPECT_TRUE(gate1.input_connections.empty());
    
    Gate gate2(123, GateType::OR, Position(5, 10), 2.5f);
    EXPECT_EQ(gate2.id, 123);
    EXPECT_EQ(gate2.type, GateType::OR);
    EXPECT_EQ(gate2.position.x, 5);
    EXPECT_EQ(gate2.position.y, 10);
    EXPECT_EQ(gate2.delay, 2.5f);
}

// Test Connection struct
TEST_F(TypesTest, ConnectionConstruction) {
    Connection conn1;
    EXPECT_EQ(conn1.from_gate_id, 0);
    EXPECT_EQ(conn1.to_gate_id, 0);
    EXPECT_EQ(conn1.to_input_index, 0);
    EXPECT_EQ(conn1.signal_strength, 1.0f);
    
    Connection conn2(10, 20, 1, 0.8f);
    EXPECT_EQ(conn2.from_gate_id, 10);
    EXPECT_EQ(conn2.to_gate_id, 20);
    EXPECT_EQ(conn2.to_input_index, 1);
    EXPECT_EQ(conn2.signal_strength, 0.8f);
}

// Test TestCase struct
TEST_F(TypesTest, TestCaseConstruction) {
    TestCase test1;
    EXPECT_TRUE(test1.inputs.empty());
    EXPECT_TRUE(test1.expected_outputs.empty());
    
    std::vector<LogicState> inputs = {LogicState::HIGH, LogicState::LOW};
    std::vector<LogicState> outputs = {LogicState::HIGH};
    
    TestCase test2(inputs, outputs);
    EXPECT_EQ(test2.inputs.size(), 2);
    EXPECT_EQ(test2.expected_outputs.size(), 1);
    EXPECT_EQ(test2.inputs[0], LogicState::HIGH);
    EXPECT_EQ(test2.inputs[1], LogicState::LOW);
    EXPECT_EQ(test2.expected_outputs[0], LogicState::HIGH);
}

// Test PerformanceMetrics struct
TEST_F(TypesTest, PerformanceMetricsConstruction) {
    PerformanceMetrics metrics1;
    EXPECT_EQ(metrics1.total_delay, 0.0f);
    EXPECT_EQ(metrics1.power_consumption, 0.0f);
    EXPECT_EQ(metrics1.area, 0.0f);
    EXPECT_EQ(metrics1.gate_count, 0);
    EXPECT_EQ(metrics1.connection_count, 0);
    
    PerformanceMetrics metrics2(5.5f, 12.3f, 100.0f, 15, 25);
    EXPECT_EQ(metrics2.total_delay, 5.5f);
    EXPECT_EQ(metrics2.power_consumption, 12.3f);
    EXPECT_EQ(metrics2.area, 100.0f);
    EXPECT_EQ(metrics2.gate_count, 15);
    EXPECT_EQ(metrics2.connection_count, 25);
}

// Test FitnessComponents struct
TEST_F(TypesTest, FitnessComponentsConstruction) {
    FitnessComponents fitness1;
    EXPECT_EQ(fitness1.correctness_weight, 1.0f);
    EXPECT_EQ(fitness1.delay_weight, 0.0f);
    EXPECT_EQ(fitness1.power_weight, 0.0f);
    EXPECT_EQ(fitness1.area_weight, 0.0f);
    
    FitnessComponents fitness2(0.8f, 0.3f, 0.2f, 0.1f);
    EXPECT_EQ(fitness2.correctness_weight, 0.8f);
    EXPECT_EQ(fitness2.delay_weight, 0.3f);
    EXPECT_EQ(fitness2.power_weight, 0.2f);
    EXPECT_EQ(fitness2.area_weight, 0.1f);
}

// Test EvolutionaryParams struct
TEST_F(TypesTest, EvolutionaryParamsConstruction) {
    EvolutionaryParams params1;
    EXPECT_EQ(params1.population_size, 100);
    EXPECT_EQ(params1.max_generations, 1000);
    EXPECT_EQ(params1.mutation_rate, 0.1f);
    EXPECT_EQ(params1.crossover_rate, 0.8f);
    EXPECT_EQ(params1.elitism_rate, 0.1f);
    EXPECT_EQ(params1.tournament_size, 3);
    EXPECT_EQ(params1.use_gpu_acceleration, true);
    
    EvolutionaryParams params2(200, 500, 0.15f, 0.9f, 0.05f, 5, false);
    EXPECT_EQ(params2.population_size, 200);
    EXPECT_EQ(params2.max_generations, 500);
    EXPECT_EQ(params2.mutation_rate, 0.15f);
    EXPECT_EQ(params2.crossover_rate, 0.9f);
    EXPECT_EQ(params2.elitism_rate, 0.05f);
    EXPECT_EQ(params2.tournament_size, 5);
    EXPECT_EQ(params2.use_gpu_acceleration, false);
}

// Test SimulationConfig struct
TEST_F(TypesTest, SimulationConfigConstruction) {
    SimulationConfig config1;
    EXPECT_EQ(config1.max_simulation_time, 1000.0f);
    EXPECT_EQ(config1.time_step, 0.1f);
    EXPECT_EQ(config1.convergence_threshold, 1e-6f);
    EXPECT_EQ(config1.max_iterations, 10000);
    EXPECT_EQ(config1.use_event_driven, true);
    
    SimulationConfig config2(2000.0f, 0.05f, 1e-8f, 20000, false);
    EXPECT_EQ(config2.max_simulation_time, 2000.0f);
    EXPECT_EQ(config2.time_step, 0.05f);
    EXPECT_EQ(config2.convergence_threshold, 1e-8f);
    EXPECT_EQ(config2.max_iterations, 20000);
    EXPECT_EQ(config2.use_event_driven, false);
}

// Test LogicState enum
TEST_F(TypesTest, LogicStateValues) {
    EXPECT_EQ(static_cast<int>(LogicState::LOW), 0);
    EXPECT_EQ(static_cast<int>(LogicState::HIGH), 1);
    EXPECT_EQ(static_cast<int>(LogicState::UNKNOWN), 2);
    EXPECT_EQ(static_cast<int>(LogicState::HIGH_IMPEDANCE), 3);
}

// Test SelectionStrategy enum
TEST_F(TypesTest, SelectionStrategyValues) {
    EXPECT_EQ(static_cast<int>(SelectionStrategy::TOURNAMENT), 0);
    EXPECT_EQ(static_cast<int>(SelectionStrategy::ROULETTE_WHEEL), 1);
    EXPECT_EQ(static_cast<int>(SelectionStrategy::RANK_BASED), 2);
    EXPECT_EQ(static_cast<int>(SelectionStrategy::ELITISM), 3);
}

// Test CrossoverType enum
TEST_F(TypesTest, CrossoverTypeValues) {
    EXPECT_EQ(static_cast<int>(CrossoverType::SINGLE_POINT), 0);
    EXPECT_EQ(static_cast<int>(CrossoverType::TWO_POINT), 1);
    EXPECT_EQ(static_cast<int>(CrossoverType::UNIFORM), 2);
    EXPECT_EQ(static_cast<int>(CrossoverType::ARITHMETIC), 3);
}

// Test MutationType enum
TEST_F(TypesTest, MutationTypeValues) {
    EXPECT_EQ(static_cast<int>(MutationType::GATE_TYPE), 0);
    EXPECT_EQ(static_cast<int>(MutationType::CONNECTION), 1);
    EXPECT_EQ(static_cast<int>(MutationType::POSITION), 2);
    EXPECT_EQ(static_cast<int>(MutationType::PARAMETER), 3);
}

// Test hash functions
TEST_F(TypesTest, PositionHash) {
    Position pos1(5, 10);
    Position pos2(5, 10);
    Position pos3(10, 5);
    
    std::hash<Position> hasher;
    EXPECT_EQ(hasher(pos1), hasher(pos2));
    EXPECT_NE(hasher(pos1), hasher(pos3));
}

TEST_F(TypesTest, GateHash) {
    Gate gate1(1, GateType::AND, Position(0, 0), 1.0f);
    Gate gate2(1, GateType::AND, Position(0, 0), 1.0f);
    Gate gate3(2, GateType::OR, Position(1, 1), 2.0f);
    
    std::hash<Gate> hasher;
    EXPECT_EQ(hasher(gate1), hasher(gate2));
    // Note: Different gates may have same hash (collision), but identical gates should have identical hash
}

TEST_F(TypesTest, ConnectionHash) {
    Connection conn1(1, 2, 0, 1.0f);
    Connection conn2(1, 2, 0, 1.0f);
    Connection conn3(2, 3, 1, 0.5f);
    
    std::hash<Connection> hasher;
    EXPECT_EQ(hasher(conn1), hasher(conn2));
    // Note: Different connections may have same hash (collision), but identical connections should have identical hash
} 