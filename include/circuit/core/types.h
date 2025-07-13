#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <unordered_set>
#include <functional>
#include <array>
#include <iostream>

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

// Position struct for 2D coordinates
struct Position {
    uint32_t x;
    uint32_t y;
    
    Position() : x(0), y(0) {}
    Position(uint32_t x_, uint32_t y_) : x(x_), y(y_) {}
    
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }
    
    bool operator!=(const Position& other) const {
        return !(*this == other);
    }
};

// Logic state enumeration
enum class LogicState : uint8_t {
    LOW = 0,
    HIGH = 1,
    UNKNOWN = 2,
    HIGH_IMPEDANCE = 3
};

// Mutation type enumeration
enum class MutationType : uint8_t {
    GATE_TYPE = 0,
    CONNECTION = 1,
    POSITION = 2,
    ACTIVATION = 3
};

// Simulation configuration
struct SimulationConfig {
    float max_time;           // Maximum simulation time (ns)
    float time_step;          // Time step size (ns)
    float convergence_threshold; // Convergence threshold
    uint32_t max_iterations;  // Maximum iterations
    bool enable_debug;        // Enable debug output
    
    SimulationConfig() : max_time(1000.0f), time_step(0.1f), convergence_threshold(1e-6f), 
                        max_iterations(10000), enable_debug(false) {}
    
    SimulationConfig(float max_t, float step, float threshold, uint32_t max_iter, bool debug)
        : max_time(max_t), time_step(step), convergence_threshold(threshold), 
          max_iterations(max_iter), enable_debug(debug) {}
};

// Simulation result
struct SimulationResult {
    bool success;
    float total_delay;
    float power_consumption;
    uint32_t gate_count;
    uint32_t switching_activity;
    std::vector<std::vector<SignalValue>> outputs;
    std::string error_message;
    
    SimulationResult() : success(false), total_delay(0.0f), power_consumption(0.0f), 
                        gate_count(0), switching_activity(0) {}
};

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
    Position position;
    std::vector<NodeId> inputs;
    std::vector<NodeId> outputs;
    std::vector<SignalValue> input_values;
    std::vector<SignalValue> output_values;
    LogicState output_state;
    float propagation_delay;
    bool is_evaluated;
    
    Gate() : id(0), type(GateType::NONE), position(0, 0), output_state(LogicState::LOW), 
             propagation_delay(0.0f), is_evaluated(false) {}
             
    Gate(GateId id_, GateType type_, Position pos, float delay)
        : id(id_), type(type_), position(pos), output_state(LogicState::LOW), 
          propagation_delay(delay), is_evaluated(false) {}
};

// Circuit grid dimensions
struct GridDimensions {
    uint32_t width;
    uint32_t height;
    uint32_t max_gates;
    
    GridDimensions() : width(0), height(0), max_gates(0) {}
    GridDimensions(uint32_t w, uint32_t h) : width(w), height(h), max_gates(w * h) {}
    
    bool is_valid_position(const Position& pos) const {
        return pos.x < width && pos.y < height;
    }
};

// Test case for circuit evaluation
struct TestCase {
    std::vector<LogicState> inputs;
    std::vector<LogicState> expected_outputs;
    
    TestCase() = default;
    TestCase(const std::vector<LogicState>& in, const std::vector<LogicState>& out)
        : inputs(in), expected_outputs(out) {}
};

// Circuit performance metrics
struct PerformanceMetrics {
    float correctness_score;      // 0.0 to 1.0
    float correctness;           // Alias for correctness_score
    float total_delay;           // nanoseconds
    float propagation_delay;     // Alias for total_delay
    uint32_t gate_count;         // number of gates used
    float power_consumption;     // watts
    float area_cost;            // arbitrary units
    uint32_t switching_activity; // number of signal transitions
    
    PerformanceMetrics() : correctness_score(0.0f), correctness(0.0f), total_delay(0.0f), 
                          propagation_delay(0.0f), gate_count(0), power_consumption(0.0f), 
                          area_cost(0.0f), switching_activity(0) {}
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
    Position position;  // position in the grid
    std::vector<ConnectionId> input_connections;   // variable number of inputs
    std::vector<ConnectionId> output_connections;  // variable number of outputs
    bool is_active;
    
    Gene() : gate_type(GateType::NONE), position(0, 0), is_active(false) {}
    
    // Equality operator for vector comparison
    bool operator==(const Gene& other) const {
        return gate_type == other.gate_type &&
               position == other.position &&
               input_connections == other.input_connections &&
               output_connections == other.output_connections &&
               is_active == other.is_active;
    }
    
    bool operator!=(const Gene& other) const {
        return !(*this == other);
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

// Add helper function for getting gate input count
inline uint32_t get_gate_input_count(GateType type) {
    switch (type) {
        case GateType::INPUT:  return 0;
        case GateType::OUTPUT: return 1;
        case GateType::NOT:
        case GateType::BUFFER: return 1;
        case GateType::AND:
        case GateType::OR:
        case GateType::XOR:
        case GateType::NAND:
        case GateType::NOR:
        case GateType::XNOR:   return 2;
        default:               return 0;
    }
}

// Utility functions for type conversion
inline SignalValue logic_state_to_signal(LogicState state) {
    switch (state) {
        case LogicState::LOW:  return 0;
        case LogicState::HIGH: return 1;
        default:               return 0;  // Unknown/High-impedance defaults to 0
    }
}

inline LogicState signal_to_logic_state(SignalValue signal) {
    return signal == 0 ? LogicState::LOW : LogicState::HIGH;
}

} // namespace circuit

// Hash function for Position (required for std::unordered_set<Position>)
namespace std {
    template<>
    struct hash<circuit::Position> {
        std::size_t operator()(const circuit::Position& pos) const {
            return std::hash<uint32_t>()(pos.x) ^ (std::hash<uint32_t>()(pos.y) << 1);
        }
    };
}