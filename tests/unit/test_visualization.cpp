#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "circuit/viz/visualization.h"
#include "circuit/core/types.h"
#include <memory>

using namespace circuit;

class VisualizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if OpenGL context can be created (headless testing)
        if (!VisualizationEngine::is_opengl_available()) {
            GTEST_SKIP() << "OpenGL not available, skipping visualization tests";
        }
        
        config.window_width = 800;
        config.window_height = 600;
        config.enable_vsync = false;
        config.enable_msaa = false;
        config.headless_mode = true;  // For testing
        
        viz_engine = std::make_unique<VisualizationEngine>(config);
    }
    
    void TearDown() override {
        viz_engine.reset();
    }
    
    VisualizationConfig config;
    std::unique_ptr<VisualizationEngine> viz_engine;
};

// Test visualization engine initialization
TEST_F(VisualizationTest, Initialization) {
    EXPECT_TRUE(viz_engine->initialize());
    EXPECT_TRUE(viz_engine->is_initialized());
    
    auto context_info = viz_engine->get_context_info();
    EXPECT_FALSE(context_info.renderer.empty());
    EXPECT_FALSE(context_info.version.empty());
    EXPECT_GT(context_info.max_texture_size, 0);
    EXPECT_GT(context_info.max_viewport_width, 0);
    EXPECT_GT(context_info.max_viewport_height, 0);
}

// Test renderer creation
TEST_F(VisualizationTest, RendererCreation) {
    ASSERT_TRUE(viz_engine->initialize());
    
    auto circuit_renderer = viz_engine->create_circuit_renderer();
    EXPECT_TRUE(circuit_renderer != nullptr);
    
    auto evolution_renderer = viz_engine->create_evolution_renderer();
    EXPECT_TRUE(evolution_renderer != nullptr);
    
    auto ui_renderer = viz_engine->create_ui_renderer();
    EXPECT_TRUE(ui_renderer != nullptr);
}

// Test circuit rendering
TEST_F(VisualizationTest, CircuitRendering) {
    ASSERT_TRUE(viz_engine->initialize());
    
    // Create a simple circuit
    GridDimensions grid(8, 8);
    Circuit circuit(grid, 2, 1);
    circuit.add_gate(GateType::AND, Position(2, 2));
    circuit.add_gate(GateType::OR, Position(4, 4));
    circuit.add_gate(GateType::INPUT, Position(0, 0));
    circuit.add_gate(GateType::OUTPUT, Position(6, 6));
    
    auto renderer = viz_engine->create_circuit_renderer();
    ASSERT_TRUE(renderer != nullptr);
    
    // Test rendering
    EXPECT_TRUE(renderer->render_circuit(circuit));
    
    // Test with different render modes
    EXPECT_TRUE(renderer->set_render_mode(RenderMode::SCHEMATIC));
    EXPECT_TRUE(renderer->render_circuit(circuit));
    
    EXPECT_TRUE(renderer->set_render_mode(RenderMode::LAYOUT));
    EXPECT_TRUE(renderer->render_circuit(circuit));
    
    EXPECT_TRUE(renderer->set_render_mode(RenderMode::SIMULATION));
    EXPECT_TRUE(renderer->render_circuit(circuit));
}

// Test evolution visualization
TEST_F(VisualizationTest, EvolutionVisualization) {
    ASSERT_TRUE(viz_engine->initialize());
    
    auto renderer = viz_engine->create_evolution_renderer();
    ASSERT_TRUE(renderer != nullptr);
    
    // Create test evolution data
    std::vector<EvolutionStatistics> stats;
    for (int i = 0; i < 10; i++) {
        EvolutionStatistics stat;
        stat.generation = i;
        stat.best_fitness = 0.5f + i * 0.05f;
        stat.average_fitness = 0.3f + i * 0.03f;
        stat.worst_fitness = 0.1f + i * 0.01f;
        stat.diversity = 0.8f - i * 0.05f;
        stats.push_back(stat);
    }
    
    // Test fitness plot
    EXPECT_TRUE(renderer->plot_fitness_evolution(stats));
    
    // Test diversity plot
    EXPECT_TRUE(renderer->plot_diversity_evolution(stats));
    
    // Test population visualization
    std::vector<Genome> population;
    GridDimensions grid(8, 8);
    std::mt19937 rng(42);
    
    for (int i = 0; i < 5; i++) {
        Genome genome(grid, 2, 1);
        genome.initialize_random(rng, 0.2f);
        genome.set_fitness(0.5f + i * 0.1f);
        population.push_back(genome);
    }
    
    EXPECT_TRUE(renderer->visualize_population(population));
}

// Test UI components
TEST_F(VisualizationTest, UIComponents) {
    ASSERT_TRUE(viz_engine->initialize());
    
    auto ui_renderer = viz_engine->create_ui_renderer();
    ASSERT_TRUE(ui_renderer != nullptr);
    
    // Test control panel
    UIState ui_state;
    ui_state.show_control_panel = true;
    ui_state.show_statistics = true;
    ui_state.show_circuit_editor = false;
    
    EXPECT_TRUE(ui_renderer->render_control_panel(ui_state));
    
    // Test statistics display
    EvolutionStatistics stats;
    stats.generation = 42;
    stats.best_fitness = 0.85f;
    stats.average_fitness = 0.65f;
    stats.worst_fitness = 0.45f;
    stats.diversity = 0.75f;
    
    EXPECT_TRUE(ui_renderer->render_statistics(stats));
    
    // Test parameter editor
    EvolutionaryParams params;
    params.population_size = 100;
    params.mutation_rate = 0.1f;
    params.crossover_rate = 0.8f;
    
    EXPECT_TRUE(ui_renderer->render_parameter_editor(params));
}

// Test camera controls
TEST_F(VisualizationTest, CameraControls) {
    ASSERT_TRUE(viz_engine->initialize());
    
    auto camera = viz_engine->get_camera();
    ASSERT_TRUE(camera != nullptr);
    
    // Test camera movement
    camera->set_position(0.0f, 0.0f, 10.0f);
    auto pos = camera->get_position();
    EXPECT_EQ(pos.x, 0.0f);
    EXPECT_EQ(pos.y, 0.0f);
    EXPECT_EQ(pos.z, 10.0f);
    
    // Test camera rotation
    camera->set_rotation(45.0f, 30.0f, 0.0f);
    auto rot = camera->get_rotation();
    EXPECT_EQ(rot.x, 45.0f);
    EXPECT_EQ(rot.y, 30.0f);
    EXPECT_EQ(rot.z, 0.0f);
    
    // Test zoom
    camera->set_zoom(2.0f);
    EXPECT_EQ(camera->get_zoom(), 2.0f);
    
    // Test view matrix
    auto view_matrix = camera->get_view_matrix();
    EXPECT_TRUE(view_matrix.size() == 16);  // 4x4 matrix
}

// Test input handling
TEST_F(VisualizationTest, InputHandling) {
    ASSERT_TRUE(viz_engine->initialize());
    
    auto input_handler = viz_engine->get_input_handler();
    ASSERT_TRUE(input_handler != nullptr);
    
    // Test mouse events
    MouseEvent mouse_event;
    mouse_event.button = MouseButton::LEFT;
    mouse_event.action = MouseAction::PRESS;
    mouse_event.x = 400;
    mouse_event.y = 300;
    
    input_handler->handle_mouse_event(mouse_event);
    
    // Test keyboard events
    KeyboardEvent key_event;
    key_event.key = Key::SPACE;
    key_event.action = KeyAction::PRESS;
    key_event.modifiers = KeyModifier::NONE;
    
    input_handler->handle_keyboard_event(key_event);
    
    // Test scroll events
    ScrollEvent scroll_event;
    scroll_event.x_offset = 0.0f;
    scroll_event.y_offset = 1.0f;
    
    input_handler->handle_scroll_event(scroll_event);
}

// Test texture management
TEST_F(VisualizationTest, TextureManagement) {
    ASSERT_TRUE(viz_engine->initialize());
    
    auto texture_manager = viz_engine->get_texture_manager();
    ASSERT_TRUE(texture_manager != nullptr);
    
    // Test texture creation
    TextureData texture_data;
    texture_data.width = 64;
    texture_data.height = 64;
    texture_data.format = TextureFormat::RGBA;
    texture_data.data = std::vector<uint8_t>(64 * 64 * 4, 255);  // White texture
    
    uint32_t texture_id = texture_manager->create_texture(texture_data);
    EXPECT_GT(texture_id, 0);
    
    // Test texture binding
    EXPECT_TRUE(texture_manager->bind_texture(texture_id));
    
    // Test texture deletion
    EXPECT_TRUE(texture_manager->delete_texture(texture_id));
}

// Test shader management
TEST_F(VisualizationTest, ShaderManagement) {
    ASSERT_TRUE(viz_engine->initialize());
    
    auto shader_manager = viz_engine->get_shader_manager();
    ASSERT_TRUE(shader_manager != nullptr);
    
    // Test shader compilation
    const char* vertex_shader = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
        }
    )";
    
    const char* fragment_shader = R"(
        #version 330 core
        out vec4 FragColor;
        uniform vec3 color;
        void main() {
            FragColor = vec4(color, 1.0);
        }
    )";
    
    uint32_t shader_id = shader_manager->create_shader(vertex_shader, fragment_shader);
    EXPECT_GT(shader_id, 0);
    
    // Test shader usage
    EXPECT_TRUE(shader_manager->use_shader(shader_id));
    
    // Test uniform setting
    EXPECT_TRUE(shader_manager->set_uniform_vec3(shader_id, "color", 1.0f, 0.0f, 0.0f));
    
    // Test shader deletion
    EXPECT_TRUE(shader_manager->delete_shader(shader_id));
}

// Test framebuffer operations
TEST_F(VisualizationTest, FramebufferOperations) {
    ASSERT_TRUE(viz_engine->initialize());
    
    auto framebuffer_manager = viz_engine->get_framebuffer_manager();
    ASSERT_TRUE(framebuffer_manager != nullptr);
    
    // Test framebuffer creation
    FramebufferSpec spec;
    spec.width = 512;
    spec.height = 512;
    spec.color_format = TextureFormat::RGBA;
    spec.depth_format = TextureFormat::DEPTH24;
    
    uint32_t framebuffer_id = framebuffer_manager->create_framebuffer(spec);
    EXPECT_GT(framebuffer_id, 0);
    
    // Test framebuffer binding
    EXPECT_TRUE(framebuffer_manager->bind_framebuffer(framebuffer_id));
    
    // Test framebuffer unbinding
    EXPECT_TRUE(framebuffer_manager->unbind_framebuffer());
    
    // Test framebuffer deletion
    EXPECT_TRUE(framebuffer_manager->delete_framebuffer(framebuffer_id));
}

// Test animation system
TEST_F(VisualizationTest, AnimationSystem) {
    ASSERT_TRUE(viz_engine->initialize());
    
    auto animation_manager = viz_engine->get_animation_manager();
    ASSERT_TRUE(animation_manager != nullptr);
    
    // Test animation creation
    AnimationSpec spec;
    spec.duration = 2.0f;
    spec.type = AnimationType::LINEAR;
    spec.loop = false;
    
    uint32_t animation_id = animation_manager->create_animation(spec);
    EXPECT_GT(animation_id, 0);
    
    // Test animation playback
    EXPECT_TRUE(animation_manager->play_animation(animation_id));
    EXPECT_TRUE(animation_manager->is_playing(animation_id));
    
    // Test animation pause
    EXPECT_TRUE(animation_manager->pause_animation(animation_id));
    EXPECT_FALSE(animation_manager->is_playing(animation_id));
    
    // Test animation stop
    EXPECT_TRUE(animation_manager->stop_animation(animation_id));
    
    // Test animation deletion
    EXPECT_TRUE(animation_manager->delete_animation(animation_id));
}

// Test render loop
TEST_F(VisualizationTest, RenderLoop) {
    ASSERT_TRUE(viz_engine->initialize());
    
    // Test single frame rendering
    EXPECT_TRUE(viz_engine->begin_frame());
    EXPECT_TRUE(viz_engine->end_frame());
    
    // Test multiple frames
    for (int i = 0; i < 5; i++) {
        EXPECT_TRUE(viz_engine->begin_frame());
        
        // Clear screen
        viz_engine->clear_screen(0.2f, 0.3f, 0.4f, 1.0f);
        
        // Render some content here...
        
        EXPECT_TRUE(viz_engine->end_frame());
    }
}

// Test performance monitoring
TEST_F(VisualizationTest, PerformanceMonitoring) {
    ASSERT_TRUE(viz_engine->initialize());
    
    // Start performance monitoring
    viz_engine->start_performance_monitoring();
    
    // Render some frames
    for (int i = 0; i < 10; i++) {
        viz_engine->begin_frame();
        viz_engine->clear_screen(0.0f, 0.0f, 0.0f, 1.0f);
        viz_engine->end_frame();
    }
    
    // Stop monitoring
    viz_engine->stop_performance_monitoring();
    
    // Get performance metrics
    auto metrics = viz_engine->get_performance_metrics();
    EXPECT_GT(metrics.frames_rendered, 0);
    EXPECT_GT(metrics.average_frame_time_ms, 0.0f);
    EXPECT_GT(metrics.average_fps, 0.0f);
}

// Test error handling
TEST_F(VisualizationTest, ErrorHandling) {
    ASSERT_TRUE(viz_engine->initialize());
    
    // Test invalid operations
    auto shader_manager = viz_engine->get_shader_manager();
    
    // Invalid shader compilation
    const char* invalid_shader = "invalid shader code";
    uint32_t invalid_id = shader_manager->create_shader(invalid_shader, invalid_shader);
    EXPECT_EQ(invalid_id, 0);
    
    // Invalid texture operations
    auto texture_manager = viz_engine->get_texture_manager();
    EXPECT_FALSE(texture_manager->bind_texture(999));  // Non-existent texture
    
    // Invalid framebuffer operations
    auto framebuffer_manager = viz_engine->get_framebuffer_manager();
    EXPECT_FALSE(framebuffer_manager->bind_framebuffer(999));  // Non-existent framebuffer
}

// Test cleanup
TEST_F(VisualizationTest, Cleanup) {
    ASSERT_TRUE(viz_engine->initialize());
    
    // Create some resources
    auto texture_manager = viz_engine->get_texture_manager();
    TextureData texture_data;
    texture_data.width = 32;
    texture_data.height = 32;
    texture_data.format = TextureFormat::RGBA;
    texture_data.data = std::vector<uint8_t>(32 * 32 * 4, 128);
    
    uint32_t texture_id = texture_manager->create_texture(texture_data);
    EXPECT_GT(texture_id, 0);
    
    // Cleanup should free all resources
    viz_engine->cleanup();
    EXPECT_FALSE(viz_engine->is_initialized());
}

// Test headless mode
TEST_F(VisualizationTest, HeadlessMode) {
    VisualizationConfig headless_config;
    headless_config.headless_mode = true;
    headless_config.window_width = 800;
    headless_config.window_height = 600;
    
    auto headless_engine = std::make_unique<VisualizationEngine>(headless_config);
    EXPECT_TRUE(headless_engine->initialize());
    
    // Should be able to render without a window
    EXPECT_TRUE(headless_engine->begin_frame());
    EXPECT_TRUE(headless_engine->end_frame());
}

// Test multi-threading safety
TEST_F(VisualizationTest, ThreadSafety) {
    ASSERT_TRUE(viz_engine->initialize());
    
    // Visualization should be thread-safe for read operations
    std::vector<std::thread> threads;
    std::atomic<int> success_count(0);
    
    for (int i = 0; i < 4; i++) {
        threads.emplace_back([this, &success_count]() {
            auto context_info = viz_engine->get_context_info();
            if (!context_info.renderer.empty()) {
                success_count++;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(success_count.load(), 4);
} 