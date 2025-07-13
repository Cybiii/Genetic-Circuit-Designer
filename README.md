# GPU-Accelerated Evolutionary Circuit Designer

A high-performance CUDA-accelerated framework for automatically designing and optimizing digital logic circuits using genetic algorithms. This project combines AI, hardware design, and GPU computing to evolve efficient digital circuits based on user-defined performance targets.

## Project Structure

```
Genetic-Circuit-Designer/
├── CMakeLists.txt              # Main build configuration
├── README.md                   # This file
├── plan.txt                    # Original project plan
│
├── include/                    # Header files
│   └── circuit/
│       ├── core/              # Core circuit types and classes
│       │   ├── types.h        # Fundamental data types
│       │   └── circuit.h      # Circuit class definition
│       ├── ga/                # Genetic algorithm components
│       │   ├── genome.h       # Genome representation
│       │   └── genetic_algorithm.h  # Main GA framework
│       ├── gpu/               # GPU/CUDA acceleration
│       │   └── gpu_simulator.h      # GPU simulator interface
│       ├── viz/               # Visualization components
│       │   └── visualization.h      # OpenGL/ImGui visualization
│       └── utils/             # Utility functions
│           └── utils.h        # Logging, profiling, math utils
│
├── src/                       # Source code implementation
│   ├── main.cpp              # Main application entry point
│   ├── CMakeLists.txt        # Source build configuration
│   ├── core/                 # Core implementation
│   │   ├── circuit.cpp
│   │   ├── circuit_simulator.cpp
│   │   ├── circuit_builder.cpp
│   │   └── test_generators.cpp
│   ├── ga/                   # Genetic algorithm implementation
│   │   ├── genome.cpp
│   │   ├── genetic_algorithm.cpp
│   │   ├── selection.cpp
│   │   ├── crossover.cpp
│   │   ├── mutation.cpp
│   │   └── fitness_evaluation.cpp
│   ├── gpu/                  # GPU implementation
│   │   ├── gpu_simulator.cpp
│   │   ├── circuit_kernels.cu
│   │   ├── genetic_kernels.cu
│   │   └── cuda_utils.cpp
│   ├── viz/                  # Visualization implementation
│   │   ├── visualization.cpp
│   │   ├── circuit_renderer.cpp
│   │   ├── evolution_visualizer.cpp
│   │   └── opengl_utils.cpp
│   └── utils/                # Utility implementations
│       ├── logger.cpp
│       ├── profiler.cpp
│       ├── file_utils.cpp
│       └── math_utils.cpp
│
├── tests/                     # Test suite
│   ├── CMakeLists.txt
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── performance/          # Performance benchmarks
│
├── examples/                  # Example programs
│   ├── CMakeLists.txt
│   ├── basic/                # Basic usage examples
│   ├── advanced/             # Advanced techniques
│   └── tutorials/            # Step-by-step tutorials
│
├── data/                     # Data files
│   ├── circuits/            # Pre-designed circuits
│   └── benchmarks/          # Benchmark datasets
│
├── external/                 # External dependencies
│   ├── imgui/               # ImGui for UI
│   ├── nlohmann/            # JSON library
│   └── glad/                # OpenGL loader
│
├── docs/                     # Documentation
│   ├── api/                 # API documentation
│   └── tutorials/           # User guides
│
└── scripts/                  # Build and utility scripts
    ├── build/               # Build automation
    └── benchmark/           # Performance testing
```

### Dependencies

- **OpenGL**: 3.3 or later
- **GLFW**: 3.3+ (for window management)
- **ImGui**: 1.80+ (for UI, included)
- **nlohmann/json**: 3.9+ (for serialization, included)
- **Google Test**: 1.10+ (for testing, auto-downloaded)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Genetic-Circuit-Designer.git
cd Genetic-Circuit-Designer
```

### 2. Install Dependencies

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install cmake build-essential nvidia-cuda-toolkit
sudo apt install libglfw3-dev libglew-dev libgl1-mesa-dev
```

#### Windows

1. Install [Visual Studio 2019+](https://visualstudio.microsoft.com/)
2. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
3. Install [CMake](https://cmake.org/download/)
4. Install [vcpkg](https://vcpkg.io/) for package management

#### macOS

```bash
brew install cmake glfw glew
# Install CUDA Toolkit from NVIDIA website
```

### 3. Build the Project

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

For Windows with Visual Studio:

```cmd
mkdir build && cd build
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release
```

### 4. Run Tests

```bash
ctest --verbose
```

## Usage

### Command Line Interface

The main application provides a comprehensive command-line interface:

```bash
# Basic usage - evolve a 4-bit adder
./circuit_designer --problem adder --bits 4

# Advanced usage with custom parameters
./circuit_designer \
  --problem multiplexer \
  --bits 3 \
  --population 200 \
  --generations 1000 \
  --mutation-rate 0.15 \
  --crossover-rate 0.85 \
  --grid 32x32 \
  --output-dir results/

# Interactive mode with visualization
./circuit_designer --interactive --problem adder --bits 4

# GPU vs CPU benchmark
./circuit_designer --benchmark --no-viz

# Disable GPU acceleration
./circuit_designer --problem adder --no-gpu
```

### Configuration Options

| Option               | Description                                   | Default |
| -------------------- | --------------------------------------------- | ------- |
| `--problem TYPE`     | Circuit type (adder, multiplexer, comparator) | adder   |
| `--bits N`           | Bit width of the circuit                      | 4       |
| `--grid WxH`         | Grid dimensions                               | 16x16   |
| `--population N`     | Population size                               | 100     |
| `--generations N`    | Maximum generations                           | 500     |
| `--mutation-rate F`  | Mutation rate (0.0-1.0)                       | 0.1     |
| `--crossover-rate F` | Crossover rate (0.0-1.0)                      | 0.8     |
| `--no-gpu`           | Disable GPU acceleration                      | false   |
| `--no-viz`           | Disable visualization                         | false   |
| `--interactive`      | Enable interactive mode                       | false   |
| `--verbose`          | Enable verbose logging                        | false   |

### Programming Interface

```cpp
#include "circuit/core/circuit.h"
#include "circuit/ga/genetic_algorithm.h"

using namespace circuit;

int main() {
    // Create a circuit for a 4-bit adder
    GridDimensions grid(16, 16);
    uint32_t inputs = 9;   // 4-bit A + 4-bit B + carry-in
    uint32_t outputs = 5;  // 4-bit sum + carry-out

    // Setup genetic algorithm
    EvolutionaryParams params;
    params.population_size = 100;
    params.max_generations = 500;
    params.use_gpu_acceleration = true;

    auto ga = create_genetic_algorithm(params, grid, inputs, outputs);

    // Generate test cases
    auto test_cases = generate_adder_test_cases(4);

    // Setup fitness weights
    FitnessComponents fitness;
    fitness.correctness_weight = 1.0f;
    fitness.delay_weight = 0.3f;
    fitness.power_weight = 0.2f;
    fitness.area_weight = 0.1f;

    // Run evolution
    std::mt19937 rng(42);
    ga->evolve(test_cases, fitness, rng);

    // Get best result
    const auto& best = ga->get_best_genome();
    auto circuit = best.to_circuit();

    return 0;
}
```

## Examples

### Basic Examples

1. **Circuit Creation** (`examples/basic/circuit_creation.cpp`)

   - Create and manipulate circuits programmatically
   - Add gates, connections, and test circuits

2. **Genetic Algorithm** (`examples/basic/genetic_algorithm.cpp`)

   - Run a simple genetic algorithm
   - Demonstrate selection, crossover, and mutation

3. **Visualization** (`examples/basic/visualization.cpp`)
   - Display circuits and evolution progress
   - Interactive circuit building

### Advanced Examples

1. **GPU Acceleration** (`examples/advanced/gpu_acceleration.cpp`)

   - Compare GPU vs CPU performance
   - Custom CUDA kernel implementation

2. **Evolution Strategies** (`examples/advanced/evolution_strategies.cpp`)

   - Island model evolution
   - Multi-objective optimization
   - Adaptive parameters

3. **Performance Benchmarking** (`examples/advanced/performance_benchmark.cpp`)
   - Comprehensive performance analysis
   - Memory usage profiling
   - Scalability testing

### Tutorial Series

1. **Hello Circuit** - Basic circuit creation and simulation
2. **Circuit Simulation** - Understanding the simulation engine
3. **Genetic Algorithm** - Setting up and running evolution
4. **GPU Acceleration** - Enabling and optimizing GPU usage
5. **Visualization** - Creating interactive visualizations

Run tutorials with:

```bash
make run_tutorials
```

## Architecture

### Core Components

1. **Circuit Representation**

   - Grid-based spatial organization
   - Efficient graph structure for connections
   - Support for various gate types

2. **Genetic Algorithm**

   - Population-based evolution
   - Tournament selection
   - Multi-point crossover
   - Adaptive mutation

3. **GPU Acceleration**

   - CUDA kernels for parallel simulation
   - Coalesced memory access patterns
   - Efficient population evaluation

4. **Visualization System**
   - OpenGL-based rendering
   - ImGui for user interface
   - Real-time evolution monitoring

### Performance Characteristics

- **Simulation Speed**: 10,000+ circuits/second on RTX 3060
- **Evolution Speed**: 100+ generations/second for 100-individual population
- **Memory Usage**: ~100MB for 1000-individual population
- **GPU Utilization**: 80-90% on supported hardware

## API Documentation

### Core Classes

#### `Circuit`

Main circuit representation with methods for:

- Gate management (`add_gate`, `remove_gate`)
- Connection management (`add_connection`, `remove_connection`)
- Simulation (`simulate`, `evaluate_performance`)
- Validation (`is_valid`, `has_cycles`)

#### `Genome`

Genetic representation with methods for:

- Genetic operations (`mutate`, `crossover`)
- Circuit conversion (`to_circuit`, `from_circuit`)
- Serialization (`save_to_file`, `load_from_file`)

#### `GeneticAlgorithm`

Main evolution framework with methods for:

- Population management (`initialize_population`)
- Evolution control (`evolve`, `evolve_single_generation`)
- Result access (`get_best_genome`, `get_population`)

#### `GPUSimulator`

GPU acceleration interface with methods for:

- Device management (`select_device`, `get_device_info`)
- Batch operations (`simulate_circuit_batch`, `evaluate_fitness_batch`)
- Memory management (`allocate_memory`, `deallocate_memory`)
