#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <memory>

namespace circuit {

// Forward declarations
struct Gate;
struct Circuit;
struct Genome;

// Basic type definitions
using GateId = uint32_t;
using NodeId = uint32_t;
using ConnectionId = uint32_t;
using TimeStep = uint32_t;
using SignalValue = uint8_t;  // 0 or 1 for digital circuits

// Gate types enumeration
enum class GateType : uint8_t {
    NONE = 0,
    INPUT,
    OUTPUT,
    AND,
    OR,
    NOT,
    XOR,
    NAND,
    NOR,
    XNOR,
    BUFFER,
    COUNT  // Total number of gate types
};

// Gate properties
struct GateProperties {
    GateType type;
    uint8_t num_inputs;
    uint8_t num_outputs;
    float propagation_delay;
    float power_consumption;
    float area_cost;
    
    GateProperties() : type(GateType::NONE), num_inputs(0), num_outputs(0), 
                      propagation_delay(0.0f), power_consumption(0.0f), area_cost(0.0f) {}
};

// Connection structure
struct Connection {
    NodeId from_node;
    NodeId to_node;
    GateId from_gate;
    GateId to_gate;
    uint8_t from_output_pin;
    uint8_t to_input_pin;
    
    Connection() : from_node(0), to_node(0), from_gate(0), to_gate(0), 
                   from_output_pin(0), to_input_pin(0) {}
};

// Gate structure
struct Gate {
    GateId id;
    GateType type;
    std::vector<NodeId> inputs;
    std::vector<NodeId> outputs;
    std::vector<SignalValue> input_values;
    std::vector<SignalValue> output_values;
    float propagation_delay;
    bool is_evaluated;
    
    Gate() : id(0), type(GateType::NONE), propagation_delay(0.0f), is_evaluated(false) {}
};

// Circuit grid dimensions
struct GridDimensions {
    uint32_t width;
    uint32_t height;
    uint32_t max_gates;
    
    GridDimensions() : width(0), height(0), max_gates(0) {}
    GridDimensions(uint32_t w, uint32_t h) : width(w), height(h), max_gates(w * h) {}
};

// Test case for circuit evaluation
struct TestCase {
    std::vector<SignalValue> inputs;
    std::vector<SignalValue> expected_outputs;
    
    TestCase() = default;
    TestCase(const std::vector<SignalValue>& in, const std::vector<SignalValue>& out)
        : inputs(in), expected_outputs(out) {}
};

// Circuit performance metrics
struct PerformanceMetrics {
    float correctness_score;      // 0.0 to 1.0
    float total_delay;           // nanoseconds
    uint32_t gate_count;         // number of gates used
    float power_consumption;     // watts
    float area_cost;            // arbitrary units
    uint32_t switching_activity; // number of signal transitions
    
    PerformanceMetrics() : correctness_score(0.0f), total_delay(0.0f), gate_count(0), 
                          power_consumption(0.0f), area_cost(0.0f), switching_activity(0) {}
};

// Fitness components for genetic algorithm
struct FitnessComponents {
    float correctness_weight;
    float delay_weight;
    float power_weight;
    float area_weight;
    
    FitnessComponents() : correctness_weight(1.0f), delay_weight(0.3f), 
                         power_weight(0.2f), area_weight(0.1f) {}
};

// Genome gene structure
struct Gene {
    GateType gate_type;
    uint32_t grid_position;  // position in the grid
    std::array<ConnectionId, 4> input_connections;  // max 4 inputs
    bool is_active;
    
    Gene() : gate_type(GateType::NONE), grid_position(0), is_active(false) {
        input_connections.fill(0);
    }
};

// Constants
constexpr uint32_t MAX_INPUTS_PER_GATE = 4;
constexpr uint32_t MAX_OUTPUTS_PER_GATE = 1;
constexpr uint32_t MAX_GRID_WIDTH = 64;
constexpr uint32_t MAX_GRID_HEIGHT = 64;
constexpr uint32_t MAX_CIRCUIT_INPUTS = 32;
constexpr uint32_t MAX_CIRCUIT_OUTPUTS = 32;
constexpr float DEFAULT_PROPAGATION_DELAY = 1.0f;  // nanoseconds

// Utility functions
inline const char* gate_type_to_string(GateType type) {
    switch (type) {
        case GateType::INPUT:  return "INPUT";
        case GateType::OUTPUT: return "OUTPUT";
        case GateType::AND:    return "AND";
        case GateType::OR:     return "OR";
        case GateType::NOT:    return "NOT";
        case GateType::XOR:    return "XOR";
        case GateType::NAND:   return "NAND";
        case GateType::NOR:    return "NOR";
        case GateType::XNOR:   return "XNOR";
        case GateType::BUFFER: return "BUFFER";
        default:               return "UNKNOWN";
    }
}

inline GateProperties get_gate_properties(GateType type) {
    GateProperties props;
    props.type = type;
    
    switch (type) {
        case GateType::INPUT:
            props.num_inputs = 0;
            props.num_outputs = 1;
            props.propagation_delay = 0.0f;
            props.power_consumption = 0.0f;
            props.area_cost = 0.5f;
            break;
            
        case GateType::OUTPUT:
            props.num_inputs = 1;
            props.num_outputs = 0;
            props.propagation_delay = 0.1f;
            props.power_consumption = 0.1f;
            props.area_cost = 0.5f;
            break;
            
        case GateType::NOT:
        case GateType::BUFFER:
            props.num_inputs = 1;
            props.num_outputs = 1;
            props.propagation_delay = 0.5f;
            props.power_consumption = 0.2f;
            props.area_cost = 1.0f;
            break;
            
        case GateType::AND:
        case GateType::OR:
        case GateType::XOR:
        case GateType::NAND:
        case GateType::NOR:
        case GateType::XNOR:
            props.num_inputs = 2;
            props.num_outputs = 1;
            props.propagation_delay = 1.0f;
            props.power_consumption = 0.4f;
            props.area_cost = 2.0f;
            break;
            
        default:
            props.num_inputs = 0;
            props.num_outputs = 0;
            props.propagation_delay = 0.0f;
            props.power_consumption = 0.0f;
            props.area_cost = 0.0f;
            break;
    }
    
    return props;
}

} // namespace circuit 