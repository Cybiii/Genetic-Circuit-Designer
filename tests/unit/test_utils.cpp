#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "circuit/utils/utils.h"
#include "circuit/core/types.h"
#include <filesystem>
#include <fstream>

using namespace circuit;

class UtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test directory
        test_dir = "test_utils_dir";
        std::filesystem::create_directories(test_dir);
    }
    
    void TearDown() override {
        // Clean up test directory
        std::filesystem::remove_all(test_dir);
    }
    
    std::string test_dir;
};

// Test Logger functionality
TEST_F(UtilsTest, LoggerBasics) {
    std::string log_file = test_dir + "/test.log";
    
    // Test logger initialization
    Logger logger(log_file);
    EXPECT_TRUE(logger.is_initialized());
    
    // Test different log levels
    logger.set_level(LogLevel::DEBUG);
    EXPECT_EQ(logger.get_level(), LogLevel::DEBUG);
    
    logger.debug("Debug message");
    logger.info("Info message");
    logger.warning("Warning message");
    logger.error("Error message");
    
    // Test log file exists
    EXPECT_TRUE(std::filesystem::exists(log_file));
    
    // Test log file contains messages
    std::ifstream file(log_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find("Debug message") != std::string::npos);
    EXPECT_TRUE(content.find("Info message") != std::string::npos);
    EXPECT_TRUE(content.find("Warning message") != std::string::npos);
    EXPECT_TRUE(content.find("Error message") != std::string::npos);
}

TEST_F(UtilsTest, LoggerLevels) {
    std::string log_file = test_dir + "/level_test.log";
    Logger logger(log_file);
    
    // Test INFO level filtering
    logger.set_level(LogLevel::INFO);
    logger.debug("Debug message");
    logger.info("Info message");
    logger.warning("Warning message");
    
    std::ifstream file(log_file);
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    
    EXPECT_TRUE(content.find("Debug message") == std::string::npos);
    EXPECT_TRUE(content.find("Info message") != std::string::npos);
    EXPECT_TRUE(content.find("Warning message") != std::string::npos);
}

// Test Profiler functionality
TEST_F(UtilsTest, ProfilerBasics) {
    Profiler profiler;
    
    // Test timer start/stop
    profiler.start_timer("test_operation");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler.stop_timer("test_operation");
    
    auto elapsed = profiler.get_elapsed_time("test_operation");
    EXPECT_GT(elapsed, 0.0);
    EXPECT_LT(elapsed, 100.0);  // Should be much less than 100ms
    
    // Test multiple operations
    profiler.start_timer("operation1");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    profiler.stop_timer("operation1");
    
    profiler.start_timer("operation2");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    profiler.stop_timer("operation2");
    
    auto stats = profiler.get_statistics();
    EXPECT_EQ(stats.size(), 3);  // test_operation, operation1, operation2
    
    EXPECT_TRUE(stats.find("test_operation") != stats.end());
    EXPECT_TRUE(stats.find("operation1") != stats.end());
    EXPECT_TRUE(stats.find("operation2") != stats.end());
}

TEST_F(UtilsTest, ProfilerScopedTimer) {
    Profiler profiler;
    
    {
        ScopedTimer timer(profiler, "scoped_operation");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    auto elapsed = profiler.get_elapsed_time("scoped_operation");
    EXPECT_GT(elapsed, 0.0);
}

TEST_F(UtilsTest, ProfilerReport) {
    Profiler profiler;
    
    // Add some test data
    profiler.start_timer("fast_op");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    profiler.stop_timer("fast_op");
    
    profiler.start_timer("slow_op");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler.stop_timer("slow_op");
    
    std::string report = profiler.generate_report();
    EXPECT_TRUE(report.find("fast_op") != std::string::npos);
    EXPECT_TRUE(report.find("slow_op") != std::string::npos);
    
    // Save report
    std::string report_file = test_dir + "/profile_report.txt";
    EXPECT_TRUE(profiler.save_report(report_file));
    EXPECT_TRUE(std::filesystem::exists(report_file));
}

// Test file utilities
TEST_F(UtilsTest, FileUtilities) {
    std::string test_file = test_dir + "/test_file.txt";
    std::string content = "Hello, World!\nThis is a test file.";
    
    // Test write file
    EXPECT_TRUE(write_file(test_file, content));
    EXPECT_TRUE(std::filesystem::exists(test_file));
    
    // Test read file
    std::string read_content;
    EXPECT_TRUE(read_file(test_file, read_content));
    EXPECT_EQ(read_content, content);
    
    // Test file exists
    EXPECT_TRUE(file_exists(test_file));
    EXPECT_FALSE(file_exists("non_existent_file.txt"));
    
    // Test file size
    size_t size = get_file_size(test_file);
    EXPECT_EQ(size, content.size());
    
    // Test directory utilities
    std::string sub_dir = test_dir + "/subdir";
    EXPECT_TRUE(create_directory(sub_dir));
    EXPECT_TRUE(directory_exists(sub_dir));
    
    // Test file listing
    auto files = list_files(test_dir);
    EXPECT_GT(files.size(), 0);
    
    bool found_test_file = false;
    for (const auto& file : files) {
        if (file.find("test_file.txt") != std::string::npos) {
            found_test_file = true;
            break;
        }
    }
    EXPECT_TRUE(found_test_file);
}

TEST_F(UtilsTest, FileUtilitiesEdgeCases) {
    // Test reading non-existent file
    std::string content;
    EXPECT_FALSE(read_file("non_existent.txt", content));
    
    // Test writing to invalid path
    EXPECT_FALSE(write_file("/invalid/path/file.txt", "content"));
    
    // Test file size of non-existent file
    EXPECT_EQ(get_file_size("non_existent.txt"), 0);
}

// Test math utilities
TEST_F(UtilsTest, MathUtilities) {
    // Test random number generation
    RandomNumberGenerator rng(42);
    
    // Test uniform distribution
    float value = rng.uniform(0.0f, 1.0f);
    EXPECT_GE(value, 0.0f);
    EXPECT_LE(value, 1.0f);
    
    // Test normal distribution
    float normal_value = rng.normal(0.0f, 1.0f);
    EXPECT_TRUE(std::isfinite(normal_value));
    
    // Test integer generation
    int int_value = rng.uniform_int(1, 10);
    EXPECT_GE(int_value, 1);
    EXPECT_LE(int_value, 10);
    
    // Test reproducibility
    RandomNumberGenerator rng2(42);
    float value2 = rng2.uniform(0.0f, 1.0f);
    EXPECT_EQ(value, value2);  // Same seed should give same result
}

TEST_F(UtilsTest, MathFunctions) {
    // Test clamp function
    EXPECT_EQ(clamp(5.0f, 0.0f, 10.0f), 5.0f);
    EXPECT_EQ(clamp(-5.0f, 0.0f, 10.0f), 0.0f);
    EXPECT_EQ(clamp(15.0f, 0.0f, 10.0f), 10.0f);
    
    // Test lerp function
    EXPECT_EQ(lerp(0.0f, 10.0f, 0.0f), 0.0f);
    EXPECT_EQ(lerp(0.0f, 10.0f, 1.0f), 10.0f);
    EXPECT_EQ(lerp(0.0f, 10.0f, 0.5f), 5.0f);
    
    // Test smoothstep function
    EXPECT_EQ(smoothstep(0.0f, 1.0f, 0.0f), 0.0f);
    EXPECT_EQ(smoothstep(0.0f, 1.0f, 1.0f), 1.0f);
    float mid = smoothstep(0.0f, 1.0f, 0.5f);
    EXPECT_GT(mid, 0.0f);
    EXPECT_LT(mid, 1.0f);
    
    // Test radians/degrees conversion
    EXPECT_FLOAT_EQ(degrees_to_radians(180.0f), M_PI);
    EXPECT_FLOAT_EQ(radians_to_degrees(M_PI), 180.0f);
    
    // Test power of 2 functions
    EXPECT_TRUE(is_power_of_2(1));
    EXPECT_TRUE(is_power_of_2(2));
    EXPECT_TRUE(is_power_of_2(4));
    EXPECT_TRUE(is_power_of_2(8));
    EXPECT_FALSE(is_power_of_2(3));
    EXPECT_FALSE(is_power_of_2(5));
    
    EXPECT_EQ(next_power_of_2(3), 4);
    EXPECT_EQ(next_power_of_2(5), 8);
    EXPECT_EQ(next_power_of_2(8), 8);
}

// Test string utilities
TEST_F(UtilsTest, StringUtilities) {
    // Test string splitting
    std::string text = "apple,banana,cherry";
    auto parts = split_string(text, ",");
    EXPECT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0], "apple");
    EXPECT_EQ(parts[1], "banana");
    EXPECT_EQ(parts[2], "cherry");
    
    // Test string joining
    std::vector<std::string> words = {"hello", "world", "test"};
    std::string joined = join_strings(words, " ");
    EXPECT_EQ(joined, "hello world test");
    
    // Test string trimming
    std::string padded = "  hello world  ";
    EXPECT_EQ(trim_string(padded), "hello world");
    
    // Test case conversion
    std::string text1 = "Hello World";
    EXPECT_EQ(to_lower(text1), "hello world");
    EXPECT_EQ(to_upper(text1), "HELLO WORLD");
    
    // Test string replacement
    std::string text2 = "Hello World World";
    EXPECT_EQ(replace_string(text2, "World", "Universe"), "Hello Universe Universe");
    
    // Test starts_with and ends_with
    std::string text3 = "Hello World";
    EXPECT_TRUE(starts_with(text3, "Hello"));
    EXPECT_FALSE(starts_with(text3, "World"));
    EXPECT_TRUE(ends_with(text3, "World"));
    EXPECT_FALSE(ends_with(text3, "Hello"));
}

// Test time utilities
TEST_F(UtilsTest, TimeUtilities) {
    // Test current time
    auto now = get_current_time();
    EXPECT_GT(now, 0);
    
    // Test formatted time
    std::string time_str = format_time(now);
    EXPECT_FALSE(time_str.empty());
    
    // Test high-resolution timer
    auto start = get_high_resolution_time();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto end = get_high_resolution_time();
    
    double elapsed = (end - start) * 1000.0;  // Convert to milliseconds
    EXPECT_GT(elapsed, 5.0);   // Should be at least 5ms
    EXPECT_LT(elapsed, 50.0);  // Should be less than 50ms
    
    // Test timer class
    Timer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.stop();
    
    EXPECT_GT(timer.elapsed_seconds(), 0.005);  // At least 5ms
    EXPECT_LT(timer.elapsed_seconds(), 0.050);  // Less than 50ms
    
    EXPECT_GT(timer.elapsed_milliseconds(), 5.0);
    EXPECT_LT(timer.elapsed_milliseconds(), 50.0);
}

// Test memory utilities
TEST_F(UtilsTest, MemoryUtilities) {
    // Test memory usage
    auto memory_info = get_memory_info();
    EXPECT_GT(memory_info.total_physical_mb, 0);
    EXPECT_GT(memory_info.available_physical_mb, 0);
    EXPECT_LE(memory_info.available_physical_mb, memory_info.total_physical_mb);
    
    // Test memory pool
    MemoryPool pool(1024);  // 1KB pool
    EXPECT_EQ(pool.get_total_size(), 1024);
    EXPECT_EQ(pool.get_used_size(), 0);
    EXPECT_EQ(pool.get_available_size(), 1024);
    
    // Allocate memory
    void* ptr1 = pool.allocate(256);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_EQ(pool.get_used_size(), 256);
    
    void* ptr2 = pool.allocate(256);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(pool.get_used_size(), 512);
    
    // Deallocate memory
    pool.deallocate(ptr1);
    EXPECT_LT(pool.get_used_size(), 512);
    
    pool.deallocate(ptr2);
    EXPECT_EQ(pool.get_used_size(), 0);
}

// Test hash utilities
TEST_F(UtilsTest, HashUtilities) {
    // Test string hashing
    std::string text1 = "Hello World";
    std::string text2 = "Hello World";
    std::string text3 = "Hello Universe";
    
    uint64_t hash1 = hash_string(text1);
    uint64_t hash2 = hash_string(text2);
    uint64_t hash3 = hash_string(text3);
    
    EXPECT_EQ(hash1, hash2);  // Same strings should have same hash
    EXPECT_NE(hash1, hash3);  // Different strings should have different hash
    
    // Test data hashing
    std::vector<uint8_t> data1 = {1, 2, 3, 4, 5};
    std::vector<uint8_t> data2 = {1, 2, 3, 4, 5};
    std::vector<uint8_t> data3 = {5, 4, 3, 2, 1};
    
    uint64_t data_hash1 = hash_data(data1.data(), data1.size());
    uint64_t data_hash2 = hash_data(data2.data(), data2.size());
    uint64_t data_hash3 = hash_data(data3.data(), data3.size());
    
    EXPECT_EQ(data_hash1, data_hash2);
    EXPECT_NE(data_hash1, data_hash3);
    
    // Test combining hashes
    uint64_t combined1 = combine_hashes(hash1, data_hash1);
    uint64_t combined2 = combine_hashes(hash1, data_hash1);
    uint64_t combined3 = combine_hashes(hash1, data_hash3);
    
    EXPECT_EQ(combined1, combined2);
    EXPECT_NE(combined1, combined3);
}

// Test threading utilities
TEST_F(UtilsTest, ThreadingUtilities) {
    // Test thread pool
    ThreadPool pool(4);
    EXPECT_EQ(pool.get_thread_count(), 4);
    
    // Test task execution
    std::atomic<int> counter(0);
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < 10; i++) {
        futures.push_back(pool.submit([&counter]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            counter++;
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    EXPECT_EQ(counter.load(), 10);
    
    // Test task with return value
    auto result_future = pool.submit([]() -> int {
        return 42;
    });
    
    int result = result_future.get();
    EXPECT_EQ(result, 42);
}

// Test system utilities
TEST_F(UtilsTest, SystemUtilities) {
    // Test system information
    auto sys_info = get_system_info();
    EXPECT_FALSE(sys_info.os_name.empty());
    EXPECT_FALSE(sys_info.cpu_name.empty());
    EXPECT_GT(sys_info.cpu_cores, 0);
    EXPECT_GT(sys_info.total_memory_mb, 0);
    
    // Test environment variables
    set_environment_variable("TEST_VAR", "test_value");
    std::string value = get_environment_variable("TEST_VAR");
    EXPECT_EQ(value, "test_value");
    
    // Test executable path
    std::string exe_path = get_executable_path();
    EXPECT_FALSE(exe_path.empty());
    
    // Test working directory
    std::string cwd = get_current_working_directory();
    EXPECT_FALSE(cwd.empty());
}

// Test error handling utilities
TEST_F(UtilsTest, ErrorHandling) {
    // Test exception handling
    try {
        throw std::runtime_error("Test error");
    } catch (const std::exception& e) {
        std::string stack_trace = get_stack_trace();
        EXPECT_FALSE(stack_trace.empty());
        
        std::string error_msg = format_exception(e);
        EXPECT_TRUE(error_msg.find("Test error") != std::string::npos);
    }
    
    // Test error code handling
    ErrorCode error = ErrorCode::SUCCESS;
    EXPECT_EQ(error_code_to_string(error), "Success");
    
    error = ErrorCode::INVALID_PARAMETER;
    EXPECT_EQ(error_code_to_string(error), "Invalid parameter");
}

// Test configuration utilities
TEST_F(UtilsTest, ConfigurationUtilities) {
    std::string config_file = test_dir + "/config.json";
    
    // Create test configuration
    nlohmann::json config;
    config["window_width"] = 800;
    config["window_height"] = 600;
    config["fullscreen"] = false;
    config["vsync"] = true;
    
    // Test saving configuration
    EXPECT_TRUE(save_config(config_file, config));
    EXPECT_TRUE(std::filesystem::exists(config_file));
    
    // Test loading configuration
    nlohmann::json loaded_config;
    EXPECT_TRUE(load_config(config_file, loaded_config));
    EXPECT_EQ(loaded_config["window_width"], 800);
    EXPECT_EQ(loaded_config["window_height"], 600);
    EXPECT_EQ(loaded_config["fullscreen"], false);
    EXPECT_EQ(loaded_config["vsync"], true);
}

// Test validation utilities
TEST_F(UtilsTest, ValidationUtilities) {
    // Test range validation
    EXPECT_TRUE(is_in_range(5, 1, 10));
    EXPECT_FALSE(is_in_range(0, 1, 10));
    EXPECT_FALSE(is_in_range(11, 1, 10));
    
    // Test email validation
    EXPECT_TRUE(is_valid_email("test@example.com"));
    EXPECT_FALSE(is_valid_email("invalid-email"));
    EXPECT_FALSE(is_valid_email("test@"));
    
    // Test URL validation
    EXPECT_TRUE(is_valid_url("https://www.example.com"));
    EXPECT_TRUE(is_valid_url("http://example.com"));
    EXPECT_FALSE(is_valid_url("invalid-url"));
    
    // Test file path validation
    EXPECT_TRUE(is_valid_file_path("/path/to/file.txt"));
    EXPECT_TRUE(is_valid_file_path("relative/path/file.txt"));
    EXPECT_FALSE(is_valid_file_path(""));
    
    // Test JSON validation
    std::string valid_json = R"({"key": "value", "number": 42})";
    std::string invalid_json = R"({"key": "value", "number": })";
    
    EXPECT_TRUE(is_valid_json(valid_json));
    EXPECT_FALSE(is_valid_json(invalid_json));
} 