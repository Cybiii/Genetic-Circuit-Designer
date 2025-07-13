#pragma once

#include "types.h"
#include <unordered_map>
#include <queue>
#include <functional>

namespace circuit {

// Forward declarations
class CircuitSimulator;

// Circuit class definition
class Circuit {
public:
    // Constructor/Destructor
    Circuit();
    Circuit(const GridDimensions& grid_dims);
    ~Circuit();
    
    // Copy/Move constructors
    Circuit(const Circuit& other);
    Circuit(Circuit&& other) noexcept;
    Circuit& operator=(const Circuit& other);
    Circuit& operator=(Circuit&& other) noexcept;
    
    // Circuit construction methods
    bool initialize(const GridDimensions& grid_dims);
    void clear();
    void reset();
    
    // Gate management
    GateId add_gate(GateType type, uint32_t grid_x, uint32_t grid_y);
    bool remove_gate(GateId gate_id);
    bool set_gate_type(GateId gate_id, GateType type);
    Gate* get_gate(GateId gate_id);
    const Gate* get_gate(GateId gate_id) const;
    
    // Connection management
    bool add_connection(GateId from_gate, uint8_t from_pin, GateId to_gate, uint8_t to_pin);
    bool remove_connection(GateId from_gate, uint8_t from_pin, GateId to_gate, uint8_t to_pin);
    bool has_connection(GateId from_gate, uint8_t from_pin, GateId to_gate, uint8_t to_pin) const;
    
    // Input/Output management
    bool add_input(uint32_t grid_x, uint32_t grid_y);
    bool add_output(uint32_t grid_x, uint32_t grid_y);
    bool set_input_value(uint32_t input_index, SignalValue value);
    SignalValue get_output_value(uint32_t output_index) const;
    
    // Circuit information
    uint32_t get_gate_count() const;
    uint32_t get_connection_count() const;
    uint32_t get_input_count() const;
    uint32_t get_output_count() const;
    const GridDimensions& get_grid_dimensions() const;
    
    // Validation
    bool is_valid() const;
    bool has_cycles() const;
    std::vector<std::string> get_validation_errors() const;
    
    // Simulation interface
    bool simulate(const std::vector<SignalValue>& inputs, std::vector<SignalValue>& outputs);
    bool simulate_test_case(const TestCase& test_case, std::vector<SignalValue>& outputs);
    PerformanceMetrics evaluate_performance(const std::vector<TestCase>& test_cases);
    
    // Utility methods
    void print_circuit() const;
    void print_gates() const;
    void print_connections() const;
    std::string to_string() const;
    
    // Serialization
    bool save_to_file(const std::string& filename) const;
    bool load_from_file(const std::string& filename);
    
    // Grid access
    GateId get_gate_at_position(uint32_t x, uint32_t y) const;
    bool is_position_occupied(uint32_t x, uint32_t y) const;
    
    // Statistics
    struct CircuitStats {
        uint32_t total_gates;
        uint32_t total_connections;
        uint32_t input_gates;
        uint32_t output_gates;
        uint32_t logic_gates;
        float total_area;
        float total_power;
        float longest_path_delay;
        uint32_t circuit_depth;
        
        CircuitStats() : total_gates(0), total_connections(0), input_gates(0),
                        output_gates(0), logic_gates(0), total_area(0.0f),
                        total_power(0.0f), longest_path_delay(0.0f), circuit_depth(0) {}
    };
    
    CircuitStats get_statistics() const;
    
private:
    // Internal data structures
    std::unordered_map<GateId, std::unique_ptr<Gate>> gates_;
    std::vector<std::vector<Connection>> connections_;  // adjacency list
    std::vector<GateId> input_gates_;
    std::vector<GateId> output_gates_;
    GridDimensions grid_dims_;
    std::vector<std::vector<GateId>> grid_;  // 2D grid for spatial organization
    
    // Internal state
    GateId next_gate_id_;
    bool is_simulation_ready_;
    
    // Helper methods
    void initialize_grid();
    void clear_grid();
    bool validate_position(uint32_t x, uint32_t y) const;
    bool validate_gate_id(GateId gate_id) const;
    bool validate_connection(GateId from_gate, uint8_t from_pin, GateId to_gate, uint8_t to_pin) const;
    
    // Topological sorting for simulation
    std::vector<GateId> topological_sort() const;
    bool dfs_has_cycle(GateId gate_id, std::unordered_set<GateId>& visited, 
                      std::unordered_set<GateId>& recursion_stack) const;
    
    // Simulation helpers
    bool propagate_signals();
    void reset_gate_states();
    void compute_gate_output(Gate* gate);
    
    // Path analysis
    float compute_longest_path_delay() const;
    uint32_t compute_circuit_depth() const;
    
    friend class CircuitSimulator;
};

// Circuit builder helper class
class CircuitBuilder {
public:
    CircuitBuilder(const GridDimensions& grid_dims);
    
    // Builder methods
    CircuitBuilder& add_gate(GateType type, uint32_t x, uint32_t y);
    CircuitBuilder& connect(uint32_t from_x, uint32_t from_y, uint32_t to_x, uint32_t to_y);
    CircuitBuilder& connect(uint32_t from_x, uint32_t from_y, uint8_t from_pin, 
                           uint32_t to_x, uint32_t to_y, uint8_t to_pin);
    CircuitBuilder& add_input(uint32_t x, uint32_t y);
    CircuitBuilder& add_output(uint32_t x, uint32_t y);
    
    // Build the circuit
    std::unique_ptr<Circuit> build();
    
private:
    GridDimensions grid_dims_;
    std::vector<std::tuple<GateType, uint32_t, uint32_t>> gates_to_add_;
    std::vector<std::tuple<uint32_t, uint32_t, uint8_t, uint32_t, uint32_t, uint8_t>> connections_to_add_;
    std::vector<std::pair<uint32_t, uint32_t>> inputs_to_add_;
    std::vector<std::pair<uint32_t, uint32_t>> outputs_to_add_;
};

// Utility functions
std::unique_ptr<Circuit> create_adder_circuit(uint32_t bit_width);
std::unique_ptr<Circuit> create_multiplexer_circuit(uint32_t select_bits);
std::unique_ptr<Circuit> create_comparator_circuit(uint32_t bit_width);
std::vector<TestCase> generate_adder_test_cases(uint32_t bit_width);
std::vector<TestCase> generate_multiplexer_test_cases(uint32_t select_bits);
std::vector<TestCase> generate_comparator_test_cases(uint32_t bit_width);

} // namespace circuit 