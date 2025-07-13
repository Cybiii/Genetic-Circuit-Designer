#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <random>

#include "circuit/core/circuit.h"
#include "circuit/core/types.h"
#include "circuit/ga/genetic_algorithm.h"
#include "circuit/ga/genome.h"
#include "circuit/gpu/gpu_simulator.h"
#include "circuit/viz/visualization.h"
#include "circuit/utils/utils.h"

using namespace circuit;

// Application configuration
struct AppConfig {
    bool use_gpu = true;
    bool enable_visualization = true;
    bool interactive_mode = false;
    std::string problem_type = "adder";
    std::string config_file = "config.json";
    std::string output_dir = "output";
    uint32_t bit_width = 4;
    uint32_t grid_width = 16;
    uint32_t grid_height = 16;
    uint32_t population_size = 100;
    uint32_t max_generations = 500;
    float mutation_rate = 0.1f;
    float crossover_rate = 0.8f;
    bool verbose = false;
    bool benchmark_mode = false;
    
    void print_usage() {
        std::cout << "Usage: circuit_designer [options]\n";
        std::cout << "Options:\n";
        std::cout << "  --problem TYPE          Circuit type to evolve (adder, multiplexer, comparator)\n";
        std::cout << "  --bits N                Bit width for the circuit (default: 4)\n";
        std::cout << "  --grid WxH              Grid dimensions (default: 16x16)\n";
        std::cout << "  --population N          Population size (default: 100)\n";
        std::cout << "  --generations N         Maximum generations (default: 500)\n";
        std::cout << "  --mutation-rate F       Mutation rate (default: 0.1)\n";
        std::cout << "  --crossover-rate F      Crossover rate (default: 0.8)\n";
        std::cout << "  --config FILE           Configuration file (default: config.json)\n";
        std::cout << "  --output-dir DIR        Output directory (default: output)\n";
        std::cout << "  --no-gpu                Disable GPU acceleration\n";
        std::cout << "  --no-viz                Disable visualization\n";
        std::cout << "  --interactive           Enable interactive mode\n";
        std::cout << "  --benchmark             Run benchmarks\n";
        std::cout << "  --verbose               Enable verbose output\n";
        std::cout << "  --help                  Show this help message\n";
    }
};

// Parse command line arguments
bool parse_arguments(int argc, char* argv[], AppConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            config.print_usage();
            return false;
        } else if (arg == "--problem" && i + 1 < argc) {
            config.problem_type = argv[++i];
        } else if (arg == "--bits" && i + 1 < argc) {
            config.bit_width = std::stoul(argv[++i]);
        } else if (arg == "--grid" && i + 1 < argc) {
            std::string grid_str = argv[++i];
            size_t x_pos = grid_str.find('x');
            if (x_pos != std::string::npos) {
                config.grid_width = std::stoul(grid_str.substr(0, x_pos));
                config.grid_height = std::stoul(grid_str.substr(x_pos + 1));
            }
        } else if (arg == "--population" && i + 1 < argc) {
            config.population_size = std::stoul(argv[++i]);
        } else if (arg == "--generations" && i + 1 < argc) {
            config.max_generations = std::stoul(argv[++i]);
        } else if (arg == "--mutation-rate" && i + 1 < argc) {
            config.mutation_rate = std::stof(argv[++i]);
        } else if (arg == "--crossover-rate" && i + 1 < argc) {
            config.crossover_rate = std::stof(argv[++i]);
        } else if (arg == "--config" && i + 1 < argc) {
            config.config_file = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--no-gpu") {
            config.use_gpu = false;
        } else if (arg == "--no-viz") {
            config.enable_visualization = false;
        } else if (arg == "--interactive") {
            config.interactive_mode = true;
        } else if (arg == "--benchmark") {
            config.benchmark_mode = true;
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

// Initialize application
bool initialize_app(const AppConfig& config) {
    // Setup logging
    auto& logger = Logger::instance();
    logger.set_log_level(config.verbose ? LogLevel::DEBUG : LogLevel::INFO);
    logger.set_console_output(true);
    
    // Create output directory
    if (!file_utils::create_directories(config.output_dir)) {
        logger.error("Failed to create output directory: " + config.output_dir);
        return false;
    }
    
    // Load configuration if exists
    auto& config_manager = ConfigManager::instance();
    if (file_utils::file_exists(config.config_file)) {
        if (!config_manager.load_config(config.config_file)) {
            logger.warning("Failed to load configuration file: " + config.config_file);
        }
    }
    
    return true;
}

// Run evolution experiment
bool run_evolution(const AppConfig& config) {
    auto& logger = Logger::instance();
    logger.info("Starting evolution experiment for " + config.problem_type + " circuit");
    
    // Setup problem parameters
    GridDimensions grid_dims(config.grid_width, config.grid_height);
    uint32_t num_inputs, num_outputs;
    
    if (config.problem_type == "adder") {
        num_inputs = config.bit_width * 2 + 1;  // A, B, and carry-in
        num_outputs = config.bit_width + 1;     // Sum and carry-out
    } else if (config.problem_type == "multiplexer") {
        uint32_t select_bits = config.bit_width;
        num_inputs = (1 << select_bits) + select_bits;  // Data inputs + select lines
        num_outputs = 1;
    } else if (config.problem_type == "comparator") {
        num_inputs = config.bit_width * 2;  // A and B
        num_outputs = 3;  // A>B, A=B, A<B
    } else {
        logger.error("Unknown problem type: " + config.problem_type);
        return false;
    }
    
    // Generate test cases
    std::vector<TestCase> test_cases;
    if (config.problem_type == "adder") {
        test_cases = generate_adder_test_cases(config.bit_width);
    } else if (config.problem_type == "multiplexer") {
        test_cases = generate_multiplexer_test_cases(config.bit_width);
    } else if (config.problem_type == "comparator") {
        test_cases = generate_comparator_test_cases(config.bit_width);
    }
    
    logger.info("Generated " + std::to_string(test_cases.size()) + " test cases");
    
    // Setup genetic algorithm
    EvolutionaryParams params;
    params.population_size = config.population_size;
    params.max_generations = config.max_generations;
    params.mutation_rate = config.mutation_rate;
    params.crossover_rate = config.crossover_rate;
    params.use_gpu_acceleration = config.use_gpu;
    
    auto ga = create_genetic_algorithm(params, grid_dims, num_inputs, num_outputs);
    if (!ga) {
        logger.error("Failed to create genetic algorithm");
        return false;
    }
    
    // Setup visualization if enabled
    std::unique_ptr<Visualization> viz;
    std::unique_ptr<EvolutionVisualizer> evo_viz;
    
    if (config.enable_visualization) {
        VisualizationConfig viz_config;
        viz_config.window_title = "Circuit Designer - " + config.problem_type;
        
        viz = create_visualization(viz_config);
        if (!viz) {
            logger.warning("Failed to create visualization, continuing without it");
        } else {
            evo_viz = std::make_unique<EvolutionVisualizer>(*viz);
            evo_viz->start_evolution_visualization(*ga);
        }
    }
    
    // Setup fitness components
    FitnessComponents fitness_weights;
    fitness_weights.correctness_weight = 1.0f;
    fitness_weights.delay_weight = 0.3f;
    fitness_weights.power_weight = 0.2f;
    fitness_weights.area_weight = 0.1f;
    
    // Setup callbacks
    ga->set_fitness_callback([&](const EvolutionStats& stats) {
        logger.info("Generation " + std::to_string(stats.generation) + 
                   ": Best fitness = " + std::to_string(stats.best_fitness));
        
        if (evo_viz) {
            evo_viz->update_visualization(stats);
        }
    });
    
    // Run evolution
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool success = ga->evolve(test_cases, fitness_weights, rng);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (!success) {
        logger.error("Evolution failed");
        return false;
    }
    
    // Report results
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    logger.info("Evolution completed in " + std::to_string(duration.count()) + " ms");
    
    const auto& best_genome = ga->get_best_genome();
    logger.info("Best fitness: " + std::to_string(best_genome.get_fitness()));
    
    // Save results
    std::string output_file = config.output_dir + "/" + config.problem_type + 
                             "_" + std::to_string(config.bit_width) + "bit_best.json";
    
    if (best_genome.save_to_file(output_file)) {
        logger.info("Best genome saved to: " + output_file);
    } else {
        logger.error("Failed to save best genome");
    }
    
    // Show final result in visualization
    if (evo_viz) {
        evo_viz->render_current_best(best_genome);
        
        // Keep visualization open until user closes it
        while (!viz->should_close()) {
            viz->poll_events();
            viz->begin_frame();
            evo_viz->render_current_best(best_genome);
            viz->end_frame();
            viz->swap_buffers();
        }
    }
    
    return true;
}

// Run benchmarks
bool run_benchmarks(const AppConfig& config) {
    auto& logger = Logger::instance();
    logger.info("Running benchmarks");
    
    // TODO: Implement comprehensive benchmarks
    // - Circuit simulation performance
    // - Genetic algorithm performance
    // - GPU vs CPU comparison
    // - Memory usage analysis
    
    return true;
}

// Interactive mode
bool run_interactive(const AppConfig& config) {
    auto& logger = Logger::instance();
    logger.info("Starting interactive mode");
    
    if (!config.enable_visualization) {
        logger.error("Interactive mode requires visualization");
        return false;
    }
    
    VisualizationConfig viz_config;
    viz_config.window_title = "Circuit Designer - Interactive Mode";
    
    auto viz = create_visualization(viz_config);
    if (!viz) {
        logger.error("Failed to create visualization");
        return false;
    }
    
    GridDimensions grid_dims(config.grid_width, config.grid_height);
    InteractiveBuilder builder(*viz);
    
    builder.start_interactive_mode();
    
    while (!viz->should_close()) {
        viz->poll_events();
        viz->begin_frame();
        builder.handle_user_input();
        viz->end_frame();
        viz->swap_buffers();
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    AppConfig config;
    
    // Parse command line arguments
    if (!parse_arguments(argc, argv, config)) {
        return 1;
    }
    
    // Initialize application
    if (!initialize_app(config)) {
        std::cerr << "Failed to initialize application" << std::endl;
        return 1;
    }
    
    auto& logger = Logger::instance();
    logger.info("Starting Genetic Circuit Designer");
    
    // Run application based on mode
    bool success = false;
    
    if (config.benchmark_mode) {
        success = run_benchmarks(config);
    } else if (config.interactive_mode) {
        success = run_interactive(config);
    } else {
        success = run_evolution(config);
    }
    
    if (success) {
        logger.info("Application completed successfully");
        return 0;
    } else {
        logger.error("Application failed");
        return 1;
    }
} 