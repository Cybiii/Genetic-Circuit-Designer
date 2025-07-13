#pragma once

#include "../core/types.h"
#include "../core/circuit.h"
#include "../ga/genome.h"
#include "../ga/genetic_algorithm.h"
#include <vector>
#include <memory>
#include <string>
#include <functional>

// Forward declarations for graphics libraries
struct GLFWwindow;
struct ImGuiIO;

namespace circuit {

// Visualization configuration
struct VisualizationConfig {
    uint32_t window_width;
    uint32_t window_height;
    std::string window_title;
    bool enable_vsync;
    bool enable_multisampling;
    uint32_t msaa_samples;
    
    VisualizationConfig() : window_width(1280), window_height(720),
                           window_title("Circuit Designer"),
                           enable_vsync(true), enable_multisampling(true),
                           msaa_samples(4) {}
};

// Color schemes
struct ColorScheme {
    struct {
        float r, g, b, a;
    } background, grid, gate_input, gate_output, gate_logic, 
      connection_wire, connection_active, text, ui_background;
    
    ColorScheme(); // Default constructor with sensible colors
};

// Circuit rendering parameters
struct CircuitRenderParams {
    float grid_spacing;
    float gate_size;
    float connection_width;
    float zoom_level;
    float pan_x, pan_y;
    bool show_grid;
    bool show_gate_labels;
    bool show_signal_values;
    bool animate_signals;
    
    CircuitRenderParams() : grid_spacing(50.0f), gate_size(30.0f),
                           connection_width(2.0f), zoom_level(1.0f),
                           pan_x(0.0f), pan_y(0.0f), show_grid(true),
                           show_gate_labels(true), show_signal_values(true),
                           animate_signals(true) {}
};

// Evolution visualization parameters
struct EvolutionRenderParams {
    uint32_t plot_width;
    uint32_t plot_height;
    uint32_t history_length;
    bool show_best_fitness;
    bool show_average_fitness;
    bool show_diversity;
    bool show_population_distribution;
    
    EvolutionRenderParams() : plot_width(400), plot_height(300),
                             history_length(100), show_best_fitness(true),
                             show_average_fitness(true), show_diversity(false),
                             show_population_distribution(false) {}
};

// Main visualization class
class Visualization {
public:
    Visualization();
    ~Visualization();
    
    // Initialization
    bool initialize(const VisualizationConfig& config);
    void cleanup();
    
    // Window management
    bool should_close() const;
    void poll_events();
    void swap_buffers();
    
    // Rendering
    void begin_frame();
    void end_frame();
    void clear_background();
    
    // Circuit visualization
    void render_circuit(const Circuit& circuit, const CircuitRenderParams& params);
    void render_genome_as_circuit(const Genome& genome, const CircuitRenderParams& params);
    void render_circuit_grid(const GridDimensions& grid_dims, const CircuitRenderParams& params);
    
    // Evolution visualization
    void render_evolution_progress(const std::vector<EvolutionStats>& history,
                                  const EvolutionRenderParams& params);
    void render_population_diversity(const GenomePopulation& population,
                                    const EvolutionRenderParams& params);
    void render_fitness_distribution(const GenomePopulation& population,
                                    const EvolutionRenderParams& params);
    
    // UI components
    void render_control_panel(bool& is_running, bool& step_mode, float& speed);
    void render_circuit_properties(const Circuit& circuit);
    void render_evolution_parameters(EvolutionaryParams& params);
    void render_performance_metrics(const PerformanceMetrics& metrics);
    
    // Interactive features
    bool handle_circuit_interaction(Circuit& circuit, const CircuitRenderParams& params);
    bool handle_zoom_pan(CircuitRenderParams& params);
    
    // Configuration
    void set_color_scheme(const ColorScheme& scheme);
    const ColorScheme& get_color_scheme() const;
    
    // Callbacks
    using CircuitClickCallback = std::function<void(uint32_t x, uint32_t y, int button)>;
    using GateClickCallback = std::function<void(GateId gate_id, int button)>;
    
    void set_circuit_click_callback(CircuitClickCallback callback);
    void set_gate_click_callback(GateClickCallback callback);
    
    // Screenshot and recording
    bool save_screenshot(const std::string& filename);
    bool start_recording(const std::string& filename, uint32_t fps = 30);
    void stop_recording();
    
private:
    // Window and context
    GLFWwindow* window_;
    ImGuiIO* imgui_io_;
    
    // Configuration
    VisualizationConfig config_;
    ColorScheme color_scheme_;
    
    // Rendering state
    uint32_t vao_, vbo_, ebo_;
    uint32_t shader_program_;
    
    // Callbacks
    CircuitClickCallback circuit_click_callback_;
    GateClickCallback gate_click_callback_;
    
    // Recording state
    bool is_recording_;
    std::string recording_filename_;
    uint32_t recording_fps_;
    
    // Helper methods
    bool init_opengl();
    bool init_imgui();
    bool create_shaders();
    void cleanup_opengl();
    void cleanup_imgui();
    
    // Rendering helpers
    void draw_grid(const GridDimensions& grid_dims, const CircuitRenderParams& params);
    void draw_gate(const Gate& gate, uint32_t x, uint32_t y, const CircuitRenderParams& params);
    void draw_connection(const Connection& conn, const CircuitRenderParams& params);
    void draw_signal_animation(const Connection& conn, float time, const CircuitRenderParams& params);
    
    // UI helpers
    void draw_plot(const std::vector<float>& data, const char* label,
                   uint32_t width, uint32_t height, float min_val, float max_val);
    void draw_histogram(const std::vector<float>& data, const char* label,
                       uint32_t width, uint32_t height, uint32_t bins);
    
    // Event handling
    void handle_mouse_input();
    void handle_keyboard_input();
    void handle_window_resize(int width, int height);
    
    // Coordinate transformations
    std::pair<float, float> screen_to_grid(float screen_x, float screen_y,
                                          const CircuitRenderParams& params);
    std::pair<float, float> grid_to_screen(float grid_x, float grid_y,
                                          const CircuitRenderParams& params);
    
    // Static callbacks for GLFW
    static void glfw_error_callback(int error, const char* description);
    static void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void glfw_mouse_callback(GLFWwindow* window, int button, int action, int mods);
    static void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
    static void glfw_resize_callback(GLFWwindow* window, int width, int height);
};

// Utility classes for specialized visualization

// Real-time evolution visualizer
class EvolutionVisualizer {
public:
    EvolutionVisualizer(Visualization& viz);
    
    void start_evolution_visualization(GeneticAlgorithm& ga);
    void update_visualization(const EvolutionStats& stats);
    void render_current_best(const Genome& best_genome);
    
private:
    Visualization& viz_;
    std::vector<EvolutionStats> history_;
    CircuitRenderParams circuit_params_;
    EvolutionRenderParams evolution_params_;
};

// Circuit comparison visualizer
class CircuitComparator {
public:
    CircuitComparator(Visualization& viz);
    
    void compare_circuits(const Circuit& circuit1, const Circuit& circuit2,
                         const std::vector<TestCase>& test_cases);
    void compare_genomes(const Genome& genome1, const Genome& genome2,
                        const std::vector<TestCase>& test_cases);
    
private:
    Visualization& viz_;
    
    void render_side_by_side(const Circuit& left, const Circuit& right);
    void render_performance_comparison(const PerformanceMetrics& left,
                                     const PerformanceMetrics& right);
};

// Interactive circuit builder
class InteractiveBuilder {
public:
    InteractiveBuilder(Visualization& viz);
    
    void start_interactive_mode();
    void handle_user_input();
    Circuit& get_current_circuit();
    
private:
    Visualization& viz_;
    std::unique_ptr<Circuit> current_circuit_;
    GridDimensions grid_dims_;
    
    enum class BuildMode {
        PLACE_GATE,
        CONNECT_GATES,
        DELETE_GATE,
        SIMULATE
    };
    
    BuildMode current_mode_;
    GateType selected_gate_type_;
    GateId selected_gate_id_;
    
    void handle_gate_placement(uint32_t x, uint32_t y);
    void handle_gate_connection(GateId from_gate, GateId to_gate);
    void handle_gate_deletion(GateId gate_id);
    void update_ui();
};

// Utility functions
std::unique_ptr<Visualization> create_visualization(const VisualizationConfig& config);

ColorScheme create_dark_theme();
ColorScheme create_light_theme();
ColorScheme create_high_contrast_theme();

bool export_circuit_to_image(const Circuit& circuit, const std::string& filename,
                             uint32_t width, uint32_t height);

bool export_evolution_plot(const std::vector<EvolutionStats>& history,
                          const std::string& filename,
                          uint32_t width, uint32_t height);

} // namespace circuit 