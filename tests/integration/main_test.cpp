#include <gtest/gtest.h>
#include <iostream>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running integration tests for Genetic Circuit Designer..." << std::endl;
    
    return RUN_ALL_TESTS();
} 