#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "circuit/core/circuit.h"
#include "circuit/core/types.h"

using namespace circuit;

class CircuitTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid = GridDimensions(8, 8);
        circuit = std::make_unique<Circuit>(grid, 2, 1);
    }
    
    void TearDown() override {
        circuit.reset();
    }
    
    GridDimensions grid;
    std::unique_ptr<Circuit> circuit;
};

// Test Circuit construction
TEST_F(CircuitTest, Construction) {
    EXPECT_EQ(circuit->get_grid_dimensions().width, 8);
    EXPECT_EQ(circuit->get_grid_dimensions().height, 8);
    EXPECT_EQ(circuit->get_input_count(), 2);
    EXPECT_EQ(circuit->get_output_count(), 1);
    EXPECT_EQ(circuit->get_gate_count(), 0);
    EXPECT_EQ(circuit->get_connection_count(), 0);
}

// Test gate management
TEST_F(CircuitTest, AddGate) {
    uint32_t gate_id = circuit->add_gate(GateType::AND, Position(2, 3));
    EXPECT_GT(gate_id, 0);
    EXPECT_EQ(circuit->get_gate_count(), 1);
    
    auto gate = circuit->get_gate(gate_id);
    EXPECT_TRUE(gate.has_value());
    EXPECT_EQ(gate->type, GateType::AND);
    EXPECT_EQ(gate->position.x, 2);
    EXPECT_EQ(gate->position.y, 3);
}

TEST_F(CircuitTest, AddGateInvalidPosition) {
    // Test adding gate outside grid boundaries
    uint32_t gate_id = circuit->add_gate(GateType::OR, Position(10, 10));
    EXPECT_EQ(gate_id, 0);  // Should fail
    EXPECT_EQ(circuit->get_gate_count(), 0);
}

TEST_F(CircuitTest, RemoveGate) {
    uint32_t gate_id = circuit->add_gate(GateType::NOT, Position(1, 1));
    EXPECT_TRUE(circuit->remove_gate(gate_id));
    EXPECT_EQ(circuit->get_gate_count(), 0);
    
    auto gate = circuit->get_gate(gate_id);
    EXPECT_FALSE(gate.has_value());
}

TEST_F(CircuitTest, RemoveNonExistentGate) {
    EXPECT_FALSE(circuit->remove_gate(999));
}

// Test connection management
TEST_F(CircuitTest, AddConnection) {
    uint32_t gate1_id = circuit->add_gate(GateType::AND, Position(1, 1));
    uint32_t gate2_id = circuit->add_gate(GateType::OR, Position(2, 2));
    
    bool success = circuit->add_connection(gate1_id, gate2_id, 0);
    EXPECT_TRUE(success);
    EXPECT_EQ(circuit->get_connection_count(), 1);
}

TEST_F(CircuitTest, AddConnectionInvalidGates) {
    uint32_t gate1_id = circuit->add_gate(GateType::AND, Position(1, 1));
    
    // Try to connect to non-existent gate
    bool success = circuit->add_connection(gate1_id, 999, 0);
    EXPECT_FALSE(success);
    EXPECT_EQ(circuit->get_connection_count(), 0);
}

TEST_F(CircuitTest, RemoveConnection) {
    uint32_t gate1_id = circuit->add_gate(GateType::AND, Position(1, 1));
    uint32_t gate2_id = circuit->add_gate(GateType::OR, Position(2, 2));
    
    circuit->add_connection(gate1_id, gate2_id, 0);
    bool success = circuit->remove_connection(gate1_id, gate2_id, 0);
    EXPECT_TRUE(success);
    EXPECT_EQ(circuit->get_connection_count(), 0);
}

// Test validation
TEST_F(CircuitTest, IsValidEmptyCircuit) {
    EXPECT_FALSE(circuit->is_valid());  // Empty circuit is invalid
}

TEST_F(CircuitTest, IsValidSimpleCircuit) {
    // Create a simple valid circuit: INPUT -> AND -> OUTPUT
    uint32_t input_id = circuit->add_gate(GateType::INPUT, Position(0, 0));
    uint32_t and_id = circuit->add_gate(GateType::AND, Position(1, 1));
    uint32_t output_id = circuit->add_gate(GateType::OUTPUT, Position(2, 2));
    
    circuit->add_connection(input_id, and_id, 0);
    circuit->add_connection(and_id, output_id, 0);
    
    EXPECT_TRUE(circuit->is_valid());
}

TEST_F(CircuitTest, HasCycles) {
    // Create a circuit with a cycle
    uint32_t gate1_id = circuit->add_gate(GateType::AND, Position(1, 1));
    uint32_t gate2_id = circuit->add_gate(GateType::OR, Position(2, 2));
    uint32_t gate3_id = circuit->add_gate(GateType::NOT, Position(3, 3));
    
    circuit->add_connection(gate1_id, gate2_id, 0);
    circuit->add_connection(gate2_id, gate3_id, 0);
    circuit->add_connection(gate3_id, gate1_id, 0);  // Creates cycle
    
    EXPECT_TRUE(circuit->has_cycles());
}

TEST_F(CircuitTest, NoCycles) {
    // Create a circuit without cycles
    uint32_t gate1_id = circuit->add_gate(GateType::AND, Position(1, 1));
    uint32_t gate2_id = circuit->add_gate(GateType::OR, Position(2, 2));
    uint32_t gate3_id = circuit->add_gate(GateType::NOT, Position(3, 3));
    
    circuit->add_connection(gate1_id, gate2_id, 0);
    circuit->add_connection(gate2_id, gate3_id, 0);
    
    EXPECT_FALSE(circuit->has_cycles());
}

// Test simulation interface
TEST_F(CircuitTest, SimulateEmptyInputs) {
    // Test simulation with empty inputs
    std::vector<LogicState> inputs;
    std::vector<LogicState> outputs;
    
    bool success = circuit->simulate(inputs, outputs);
    EXPECT_FALSE(success);  // Should fail with empty inputs
}

TEST_F(CircuitTest, SimulateWrongInputSize) {
    // Test simulation with wrong number of inputs
    std::vector<LogicState> inputs = {LogicState::HIGH};  // Only 1 input, but circuit expects 2
    std::vector<LogicState> outputs;
    
    bool success = circuit->simulate(inputs, outputs);
    EXPECT_FALSE(success);  // Should fail with wrong input size
}

// Test performance evaluation
TEST_F(CircuitTest, EvaluatePerformance) {
    // Add some gates to test performance evaluation
    uint32_t gate1_id = circuit->add_gate(GateType::AND, Position(1, 1));
    uint32_t gate2_id = circuit->add_gate(GateType::OR, Position(2, 2));
    circuit->add_connection(gate1_id, gate2_id, 0);
    
    auto metrics = circuit->evaluate_performance();
    EXPECT_GT(metrics.gate_count, 0);
    EXPECT_GT(metrics.connection_count, 0);
    EXPECT_GE(metrics.total_delay, 0.0f);
    EXPECT_GE(metrics.power_consumption, 0.0f);
    EXPECT_GE(metrics.area, 0.0f);
}

// Test serialization
TEST_F(CircuitTest, ToJson) {
    uint32_t gate_id = circuit->add_gate(GateType::AND, Position(1, 1));
    
    auto json = circuit->to_json();
    EXPECT_TRUE(json.contains("grid_dimensions"));
    EXPECT_TRUE(json.contains("input_count"));
    EXPECT_TRUE(json.contains("output_count"));
    EXPECT_TRUE(json.contains("gates"));
    EXPECT_TRUE(json.contains("connections"));
}

TEST_F(CircuitTest, FromJson) {
    // Create a circuit and serialize it
    uint32_t gate_id = circuit->add_gate(GateType::AND, Position(1, 1));
    auto json = circuit->to_json();
    
    // Create a new circuit from JSON
    auto new_circuit = Circuit::from_json(json);
    EXPECT_TRUE(new_circuit != nullptr);
    EXPECT_EQ(new_circuit->get_gate_count(), circuit->get_gate_count());
    EXPECT_EQ(new_circuit->get_connection_count(), circuit->get_connection_count());
}

// Test gate access
TEST_F(CircuitTest, GetAllGates) {
    uint32_t gate1_id = circuit->add_gate(GateType::AND, Position(1, 1));
    uint32_t gate2_id = circuit->add_gate(GateType::OR, Position(2, 2));
    
    auto gates = circuit->get_all_gates();
    EXPECT_EQ(gates.size(), 2);
    
    bool found_gate1 = false, found_gate2 = false;
    for (const auto& gate : gates) {
        if (gate.id == gate1_id) found_gate1 = true;
        if (gate.id == gate2_id) found_gate2 = true;
    }
    EXPECT_TRUE(found_gate1);
    EXPECT_TRUE(found_gate2);
}

TEST_F(CircuitTest, GetAllConnections) {
    uint32_t gate1_id = circuit->add_gate(GateType::AND, Position(1, 1));
    uint32_t gate2_id = circuit->add_gate(GateType::OR, Position(2, 2));
    circuit->add_connection(gate1_id, gate2_id, 0);
    
    auto connections = circuit->get_all_connections();
    EXPECT_EQ(connections.size(), 1);
    EXPECT_EQ(connections[0].from_gate_id, gate1_id);
    EXPECT_EQ(connections[0].to_gate_id, gate2_id);
    EXPECT_EQ(connections[0].to_input_index, 0);
}

// Test circuit clearing
TEST_F(CircuitTest, Clear) {
    uint32_t gate1_id = circuit->add_gate(GateType::AND, Position(1, 1));
    uint32_t gate2_id = circuit->add_gate(GateType::OR, Position(2, 2));
    circuit->add_connection(gate1_id, gate2_id, 0);
    
    circuit->clear();
    EXPECT_EQ(circuit->get_gate_count(), 0);
    EXPECT_EQ(circuit->get_connection_count(), 0);
}

// Test circuit cloning
TEST_F(CircuitTest, Clone) {
    uint32_t gate1_id = circuit->add_gate(GateType::AND, Position(1, 1));
    uint32_t gate2_id = circuit->add_gate(GateType::OR, Position(2, 2));
    circuit->add_connection(gate1_id, gate2_id, 0);
    
    auto cloned = circuit->clone();
    EXPECT_TRUE(cloned != nullptr);
    EXPECT_EQ(cloned->get_gate_count(), circuit->get_gate_count());
    EXPECT_EQ(cloned->get_connection_count(), circuit->get_connection_count());
    EXPECT_EQ(cloned->get_grid_dimensions().width, circuit->get_grid_dimensions().width);
    EXPECT_EQ(cloned->get_grid_dimensions().height, circuit->get_grid_dimensions().height);
}

// Test edge cases
TEST_F(CircuitTest, MaxGateCapacity) {
    // Test adding gates up to capacity
    uint32_t max_gates = grid.width * grid.height;
    std::vector<uint32_t> gate_ids;
    
    for (uint32_t i = 0; i < max_gates; i++) {
        uint32_t x = i % grid.width;
        uint32_t y = i / grid.width;
        uint32_t gate_id = circuit->add_gate(GateType::AND, Position(x, y));
        if (gate_id > 0) {
            gate_ids.push_back(gate_id);
        }
    }
    
    EXPECT_GT(gate_ids.size(), 0);
    EXPECT_LE(gate_ids.size(), max_gates);
}

TEST_F(CircuitTest, DuplicatePositions) {
    // Test adding multiple gates at same position
    uint32_t gate1_id = circuit->add_gate(GateType::AND, Position(1, 1));
    uint32_t gate2_id = circuit->add_gate(GateType::OR, Position(1, 1));
    
    EXPECT_GT(gate1_id, 0);
    EXPECT_EQ(gate2_id, 0);  // Should fail - position occupied
}

TEST_F(CircuitTest, SelfConnection) {
    // Test connecting gate to itself
    uint32_t gate_id = circuit->add_gate(GateType::AND, Position(1, 1));
    
    bool success = circuit->add_connection(gate_id, gate_id, 0);
    EXPECT_FALSE(success);  // Should fail - self-connection not allowed
} 