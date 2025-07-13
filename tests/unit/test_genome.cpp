#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "circuit/ga/genome.h"
#include "circuit/core/types.h"
#include <random>

using namespace circuit;

class GenomeTest : public ::testing::Test {
protected:
    void SetUp() override {
        grid = GridDimensions(8, 8);
        rng.seed(42);  // Fixed seed for reproducible tests
    }
    
    void TearDown() override {
        // Cleanup
    }
    
    GridDimensions grid;
    std::mt19937 rng;
};

// Test Genome construction
TEST_F(GenomeTest, Construction) {
    Genome genome(grid, 2, 1);
    EXPECT_EQ(genome.get_grid_dimensions().width, 8);
    EXPECT_EQ(genome.get_grid_dimensions().height, 8);
    EXPECT_EQ(genome.get_input_count(), 2);
    EXPECT_EQ(genome.get_output_count(), 1);
    EXPECT_EQ(genome.get_fitness(), 0.0f);
    EXPECT_FALSE(genome.is_evaluated());
}

// Test random initialization
TEST_F(GenomeTest, RandomInitialization) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.3f);  // 30% fill rate
    
    EXPECT_GT(genome.get_gene_count(), 0);
    EXPECT_TRUE(genome.is_valid());
}

// Test gene management
TEST_F(GenomeTest, AddGene) {
    Genome genome(grid, 2, 1);
    
    Gene gene;
    gene.gate_type = GateType::AND;
    gene.position = Position(2, 3);
    gene.delay = 1.5f;
    gene.input_connections = {1, 2};
    
    genome.add_gene(gene);
    EXPECT_EQ(genome.get_gene_count(), 1);
    
    auto retrieved_gene = genome.get_gene(0);
    EXPECT_TRUE(retrieved_gene.has_value());
    EXPECT_EQ(retrieved_gene->gate_type, GateType::AND);
    EXPECT_EQ(retrieved_gene->position.x, 2);
    EXPECT_EQ(retrieved_gene->position.y, 3);
}

TEST_F(GenomeTest, RemoveGene) {
    Genome genome(grid, 2, 1);
    
    Gene gene;
    gene.gate_type = GateType::OR;
    gene.position = Position(1, 1);
    genome.add_gene(gene);
    
    EXPECT_TRUE(genome.remove_gene(0));
    EXPECT_EQ(genome.get_gene_count(), 0);
}

TEST_F(GenomeTest, RemoveInvalidGene) {
    Genome genome(grid, 2, 1);
    EXPECT_FALSE(genome.remove_gene(999));  // Non-existent gene
}

// Test mutation
TEST_F(GenomeTest, MutateGateType) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    
    float original_fitness = genome.get_fitness();
    genome.mutate(rng, 1.0f, MutationType::GATE_TYPE);
    
    // After mutation, fitness should be reset
    EXPECT_EQ(genome.get_fitness(), 0.0f);
    EXPECT_FALSE(genome.is_evaluated());
}

TEST_F(GenomeTest, MutateConnection) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    
    genome.mutate(rng, 1.0f, MutationType::CONNECTION);
    EXPECT_FALSE(genome.is_evaluated());
}

TEST_F(GenomeTest, MutatePosition) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    
    genome.mutate(rng, 1.0f, MutationType::POSITION);
    EXPECT_FALSE(genome.is_evaluated());
}

TEST_F(GenomeTest, MutateParameter) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    
    genome.mutate(rng, 1.0f, MutationType::PARAMETER);
    EXPECT_FALSE(genome.is_evaluated());
}

// Test crossover
TEST_F(GenomeTest, SinglePointCrossover) {
    Genome parent1(grid, 2, 1);
    Genome parent2(grid, 2, 1);
    
    parent1.initialize_random(rng, 0.3f);
    parent2.initialize_random(rng, 0.3f);
    
    auto offspring = parent1.crossover(parent2, rng, CrossoverType::SINGLE_POINT);
    EXPECT_EQ(offspring.size(), 2);
    
    for (auto& child : offspring) {
        EXPECT_EQ(child.get_grid_dimensions().width, grid.width);
        EXPECT_EQ(child.get_grid_dimensions().height, grid.height);
        EXPECT_EQ(child.get_input_count(), 2);
        EXPECT_EQ(child.get_output_count(), 1);
        EXPECT_FALSE(child.is_evaluated());
    }
}

TEST_F(GenomeTest, TwoPointCrossover) {
    Genome parent1(grid, 2, 1);
    Genome parent2(grid, 2, 1);
    
    parent1.initialize_random(rng, 0.3f);
    parent2.initialize_random(rng, 0.3f);
    
    auto offspring = parent1.crossover(parent2, rng, CrossoverType::TWO_POINT);
    EXPECT_EQ(offspring.size(), 2);
}

TEST_F(GenomeTest, UniformCrossover) {
    Genome parent1(grid, 2, 1);
    Genome parent2(grid, 2, 1);
    
    parent1.initialize_random(rng, 0.3f);
    parent2.initialize_random(rng, 0.3f);
    
    auto offspring = parent1.crossover(parent2, rng, CrossoverType::UNIFORM);
    EXPECT_EQ(offspring.size(), 2);
}

TEST_F(GenomeTest, ArithmeticCrossover) {
    Genome parent1(grid, 2, 1);
    Genome parent2(grid, 2, 1);
    
    parent1.initialize_random(rng, 0.3f);
    parent2.initialize_random(rng, 0.3f);
    
    auto offspring = parent1.crossover(parent2, rng, CrossoverType::ARITHMETIC);
    EXPECT_EQ(offspring.size(), 2);
}

// Test fitness management
TEST_F(GenomeTest, FitnessManagement) {
    Genome genome(grid, 2, 1);
    
    EXPECT_EQ(genome.get_fitness(), 0.0f);
    EXPECT_FALSE(genome.is_evaluated());
    
    genome.set_fitness(0.85f);
    EXPECT_EQ(genome.get_fitness(), 0.85f);
    EXPECT_TRUE(genome.is_evaluated());
    
    genome.invalidate_fitness();
    EXPECT_EQ(genome.get_fitness(), 0.0f);
    EXPECT_FALSE(genome.is_evaluated());
}

// Test validation
TEST_F(GenomeTest, ValidationEmpty) {
    Genome genome(grid, 2, 1);
    EXPECT_FALSE(genome.is_valid());  // Empty genome is invalid
}

TEST_F(GenomeTest, ValidationValid) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    
    // After random initialization, genome should be valid
    EXPECT_TRUE(genome.is_valid());
}

TEST_F(GenomeTest, ValidationInvalidPosition) {
    Genome genome(grid, 2, 1);
    
    Gene gene;
    gene.gate_type = GateType::AND;
    gene.position = Position(10, 10);  // Outside grid
    genome.add_gene(gene);
    
    EXPECT_FALSE(genome.is_valid());
}

// Test circuit conversion
TEST_F(GenomeTest, ToCircuit) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    
    auto circuit = genome.to_circuit();
    EXPECT_TRUE(circuit != nullptr);
    EXPECT_EQ(circuit->get_grid_dimensions().width, grid.width);
    EXPECT_EQ(circuit->get_grid_dimensions().height, grid.height);
    EXPECT_EQ(circuit->get_input_count(), 2);
    EXPECT_EQ(circuit->get_output_count(), 1);
}

TEST_F(GenomeTest, FromCircuit) {
    // Create a circuit
    Circuit circuit(grid, 2, 1);
    circuit.add_gate(GateType::AND, Position(1, 1));
    circuit.add_gate(GateType::OR, Position(2, 2));
    
    // Convert to genome
    auto genome = Genome::from_circuit(circuit);
    EXPECT_TRUE(genome != nullptr);
    EXPECT_EQ(genome->get_grid_dimensions().width, grid.width);
    EXPECT_EQ(genome->get_grid_dimensions().height, grid.height);
    EXPECT_EQ(genome->get_input_count(), 2);
    EXPECT_EQ(genome->get_output_count(), 1);
}

// Test serialization
TEST_F(GenomeTest, ToJson) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    genome.set_fitness(0.75f);
    
    auto json = genome.to_json();
    EXPECT_TRUE(json.contains("grid_dimensions"));
    EXPECT_TRUE(json.contains("input_count"));
    EXPECT_TRUE(json.contains("output_count"));
    EXPECT_TRUE(json.contains("genes"));
    EXPECT_TRUE(json.contains("fitness"));
    EXPECT_TRUE(json.contains("evaluated"));
}

TEST_F(GenomeTest, FromJson) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    genome.set_fitness(0.65f);
    
    auto json = genome.to_json();
    auto new_genome = Genome::from_json(json);
    
    EXPECT_TRUE(new_genome != nullptr);
    EXPECT_EQ(new_genome->get_grid_dimensions().width, genome.get_grid_dimensions().width);
    EXPECT_EQ(new_genome->get_grid_dimensions().height, genome.get_grid_dimensions().height);
    EXPECT_EQ(new_genome->get_input_count(), genome.get_input_count());
    EXPECT_EQ(new_genome->get_output_count(), genome.get_output_count());
    EXPECT_EQ(new_genome->get_fitness(), genome.get_fitness());
    EXPECT_EQ(new_genome->is_evaluated(), genome.is_evaluated());
}

// Test file I/O
TEST_F(GenomeTest, SaveToFile) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    genome.set_fitness(0.8f);
    
    std::string filename = "test_genome.json";
    EXPECT_TRUE(genome.save_to_file(filename));
    
    // Clean up
    std::remove(filename.c_str());
}

TEST_F(GenomeTest, LoadFromFile) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    genome.set_fitness(0.9f);
    
    std::string filename = "test_genome_load.json";
    EXPECT_TRUE(genome.save_to_file(filename));
    
    auto loaded_genome = Genome::load_from_file(filename);
    EXPECT_TRUE(loaded_genome != nullptr);
    EXPECT_EQ(loaded_genome->get_fitness(), genome.get_fitness());
    
    // Clean up
    std::remove(filename.c_str());
}

// Test cloning
TEST_F(GenomeTest, Clone) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    genome.set_fitness(0.7f);
    
    auto cloned = genome.clone();
    EXPECT_TRUE(cloned != nullptr);
    EXPECT_EQ(cloned->get_gene_count(), genome.get_gene_count());
    EXPECT_EQ(cloned->get_fitness(), genome.get_fitness());
    EXPECT_EQ(cloned->is_evaluated(), genome.is_evaluated());
    
    // Verify they are independent
    cloned->set_fitness(0.5f);
    EXPECT_NE(cloned->get_fitness(), genome.get_fitness());
}

// Test comparison operators
TEST_F(GenomeTest, ComparisonOperators) {
    Genome genome1(grid, 2, 1);
    Genome genome2(grid, 2, 1);
    
    genome1.set_fitness(0.8f);
    genome2.set_fitness(0.6f);
    
    EXPECT_TRUE(genome1 > genome2);
    EXPECT_FALSE(genome1 < genome2);
    EXPECT_FALSE(genome1 == genome2);
    EXPECT_TRUE(genome1 != genome2);
    
    genome2.set_fitness(0.8f);
    EXPECT_FALSE(genome1 > genome2);
    EXPECT_FALSE(genome1 < genome2);
    EXPECT_TRUE(genome1 == genome2);
    EXPECT_FALSE(genome1 != genome2);
}

// Test gene access
TEST_F(GenomeTest, GetAllGenes) {
    Genome genome(grid, 2, 1);
    
    Gene gene1;
    gene1.gate_type = GateType::AND;
    gene1.position = Position(1, 1);
    genome.add_gene(gene1);
    
    Gene gene2;
    gene2.gate_type = GateType::OR;
    gene2.position = Position(2, 2);
    genome.add_gene(gene2);
    
    auto genes = genome.get_all_genes();
    EXPECT_EQ(genes.size(), 2);
    EXPECT_EQ(genes[0].gate_type, GateType::AND);
    EXPECT_EQ(genes[1].gate_type, GateType::OR);
}

// Test genome clearing
TEST_F(GenomeTest, Clear) {
    Genome genome(grid, 2, 1);
    genome.initialize_random(rng, 0.2f);
    genome.set_fitness(0.5f);
    
    genome.clear();
    EXPECT_EQ(genome.get_gene_count(), 0);
    EXPECT_EQ(genome.get_fitness(), 0.0f);
    EXPECT_FALSE(genome.is_evaluated());
}

// Test size limits
TEST_F(GenomeTest, MaxGeneCount) {
    Genome genome(grid, 2, 1);
    
    // Try to add more genes than grid positions
    uint32_t max_genes = grid.width * grid.height;
    uint32_t added_genes = 0;
    
    for (uint32_t i = 0; i < max_genes + 10; i++) {
        Gene gene;
        gene.gate_type = GateType::AND;
        gene.position = Position(i % grid.width, i / grid.width);
        
        if (genome.add_gene(gene)) {
            added_genes++;
        }
    }
    
    EXPECT_LE(added_genes, max_genes);
}

// Test edge cases
TEST_F(GenomeTest, EmptyGenomeMutation) {
    Genome genome(grid, 2, 1);
    
    // Mutating empty genome should not crash
    genome.mutate(rng, 0.5f, MutationType::GATE_TYPE);
    EXPECT_EQ(genome.get_gene_count(), 0);
}

TEST_F(GenomeTest, EmptyGenomeCrossover) {
    Genome parent1(grid, 2, 1);
    Genome parent2(grid, 2, 1);
    
    // Crossover of empty genomes should produce empty offspring
    auto offspring = parent1.crossover(parent2, rng, CrossoverType::SINGLE_POINT);
    EXPECT_EQ(offspring.size(), 2);
    EXPECT_EQ(offspring[0].get_gene_count(), 0);
    EXPECT_EQ(offspring[1].get_gene_count(), 0);
} 