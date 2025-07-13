#pragma once

#include "../core/types.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <fstream>
#include <unordered_map>

namespace circuit {

// Logging utilities
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4
};

class Logger {
public:
    static Logger& instance();
    
    void log(LogLevel level, const std::string& message);
    void debug(const std::string& message) { log(LogLevel::DEBUG, message); }
    void info(const std::string& message) { log(LogLevel::INFO, message); }
    void warning(const std::string& message) { log(LogLevel::WARNING, message); }
    void error(const std::string& message) { log(LogLevel::ERROR, message); }
    void critical(const std::string& message) { log(LogLevel::CRITICAL, message); }
    
    void set_log_level(LogLevel level) { min_level_ = level; }
    void set_output_file(const std::string& filename);
    void set_console_output(bool enabled) { console_output_ = enabled; }
    
private:
    Logger() = default;
    LogLevel min_level_ = LogLevel::INFO;
    std::unique_ptr<std::ofstream> log_file_;
    bool console_output_ = true;
    
    std::string level_to_string(LogLevel level);
    std::string get_timestamp();
};

// Profiling utilities
class Profiler {
public:
    static Profiler& instance();
    
    void start_timer(const std::string& name);
    void end_timer(const std::string& name);
    void log_timer(const std::string& name, const std::string& message = "");
    
    struct ProfileData {
        std::string name;
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::duration<double> total_time;
        uint32_t call_count;
        
        ProfileData() : total_time(0), call_count(0) {}
    };
    
    const std::unordered_map<std::string, ProfileData>& get_profile_data() const;
    void reset_profile_data();
    void print_profile_report();
    
private:
    Profiler() = default;
    std::unordered_map<std::string, ProfileData> profile_data_;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> active_timers_;
};

// RAII timer for easy profiling
class ScopedTimer {
public:
    ScopedTimer(const std::string& name) : name_(name) {
        Profiler::instance().start_timer(name_);
    }
    
    ~ScopedTimer() {
        Profiler::instance().end_timer(name_);
    }
    
private:
    std::string name_;
};

// Macro for easy profiling
#define PROFILE_SCOPE(name) ScopedTimer _timer(name)
#define PROFILE_FUNCTION() ScopedTimer _timer(__FUNCTION__)

// File I/O utilities
namespace file_utils {
    bool file_exists(const std::string& filename);
    bool directory_exists(const std::string& dirname);
    bool create_directory(const std::string& dirname);
    bool create_directories(const std::string& path);
    
    std::vector<std::string> list_files(const std::string& directory, 
                                       const std::string& extension = "");
    
    std::string read_file(const std::string& filename);
    bool write_file(const std::string& filename, const std::string& content);
    
    std::string get_file_extension(const std::string& filename);
    std::string get_filename_without_extension(const std::string& filename);
    std::string get_directory_path(const std::string& filename);
    
    size_t get_file_size(const std::string& filename);
    std::chrono::system_clock::time_point get_file_modification_time(const std::string& filename);
}

// String utilities
namespace string_utils {
    std::vector<std::string> split(const std::string& str, char delimiter);
    std::string trim(const std::string& str);
    std::string to_lower(const std::string& str);
    std::string to_upper(const std::string& str);
    
    bool starts_with(const std::string& str, const std::string& prefix);
    bool ends_with(const std::string& str, const std::string& suffix);
    
    std::string replace_all(const std::string& str, const std::string& from, 
                           const std::string& to);
    
    std::string format(const char* format, ...);
    
    // Number formatting
    std::string format_number(double value, int precision = 2);
    std::string format_bytes(size_t bytes);
    std::string format_duration(std::chrono::duration<double> duration);
}

// Math utilities
namespace math_utils {
    template<typename T>
    T clamp(T value, T min_val, T max_val) {
        return std::max(min_val, std::min(value, max_val));
    }
    
    template<typename T>
    T lerp(T a, T b, float t) {
        return a + t * (b - a);
    }
    
    float random_float(float min_val, float max_val);
    int random_int(int min_val, int max_val);
    
    // Statistical functions
    float mean(const std::vector<float>& values);
    float variance(const std::vector<float>& values);
    float standard_deviation(const std::vector<float>& values);
    float median(std::vector<float> values);  // Note: modifies input
    
    // Probability distributions
    float normal_distribution(float mean, float std_dev);
    float uniform_distribution(float min_val, float max_val);
    
    // Geometric utilities
    struct Point2D {
        float x, y;
        Point2D() : x(0.0f), y(0.0f) {}
        Point2D(float x_, float y_) : x(x_), y(y_) {}
    };
    
    struct Rect2D {
        float x, y, width, height;
        Rect2D() : x(0.0f), y(0.0f), width(0.0f), height(0.0f) {}
        Rect2D(float x_, float y_, float w_, float h_) : x(x_), y(y_), width(w_), height(h_) {}
        
        bool contains(const Point2D& point) const;
        bool intersects(const Rect2D& other) const;
    };
    
    float distance(const Point2D& a, const Point2D& b);
    float distance_squared(const Point2D& a, const Point2D& b);
}

// Memory utilities
namespace memory_utils {
    template<typename T>
    class MemoryPool {
    public:
        MemoryPool(size_t pool_size);
        ~MemoryPool();
        
        T* allocate();
        void deallocate(T* ptr);
        
        size_t get_available_count() const;
        size_t get_total_count() const;
        
    private:
        std::vector<T> pool_;
        std::vector<T*> free_list_;
        size_t next_free_index_;
    };
    
    // Memory tracking
    class MemoryTracker {
    public:
        static MemoryTracker& instance();
        
        void track_allocation(void* ptr, size_t size, const std::string& tag = "");
        void track_deallocation(void* ptr);
        
        size_t get_total_allocated() const;
        size_t get_peak_allocated() const;
        void print_memory_report() const;
        
    private:
        MemoryTracker() = default;
        
        struct AllocationInfo {
            size_t size;
            std::string tag;
            std::chrono::high_resolution_clock::time_point timestamp;
        };
        
        std::unordered_map<void*, AllocationInfo> allocations_;
        size_t total_allocated_ = 0;
        size_t peak_allocated_ = 0;
    };
}

// Configuration utilities
class ConfigManager {
public:
    static ConfigManager& instance();
    
    bool load_config(const std::string& filename);
    bool save_config(const std::string& filename);
    
    // Generic getters/setters
    template<typename T>
    T get(const std::string& key, const T& default_value = T{});
    
    template<typename T>
    void set(const std::string& key, const T& value);
    
    bool has_key(const std::string& key) const;
    void remove_key(const std::string& key);
    
    std::vector<std::string> get_keys() const;
    std::vector<std::string> get_keys_with_prefix(const std::string& prefix) const;
    
private:
    ConfigManager() = default;
    std::unordered_map<std::string, std::string> config_data_;
    
    std::string serialize_value(const std::string& value) const;
    std::string serialize_value(int value) const;
    std::string serialize_value(float value) const;
    std::string serialize_value(bool value) const;
    
    template<typename T>
    T deserialize_value(const std::string& str) const;
};

// Error handling utilities
class ErrorHandler {
public:
    enum class ErrorCode {
        SUCCESS = 0,
        GENERAL_ERROR,
        FILE_NOT_FOUND,
        INVALID_INPUT,
        MEMORY_ERROR,
        CUDA_ERROR,
        OPENGL_ERROR,
        NETWORK_ERROR
    };
    
    struct ErrorInfo {
        ErrorCode code;
        std::string message;
        std::string file;
        int line;
        std::chrono::high_resolution_clock::time_point timestamp;
        
        ErrorInfo(ErrorCode c, const std::string& msg, const std::string& f, int l)
            : code(c), message(msg), file(f), line(l), 
              timestamp(std::chrono::high_resolution_clock::now()) {}
    };
    
    static void report_error(ErrorCode code, const std::string& message,
                           const std::string& file = "", int line = 0);
    
    static const std::vector<ErrorInfo>& get_error_history();
    static void clear_error_history();
    
    static std::string error_code_to_string(ErrorCode code);
    
private:
    static std::vector<ErrorInfo> error_history_;
};

// Macro for easy error reporting
#define REPORT_ERROR(code, message) \
    ErrorHandler::report_error(code, message, __FILE__, __LINE__)

// Threading utilities
namespace thread_utils {
    class ThreadPool {
    public:
        ThreadPool(size_t num_threads);
        ~ThreadPool();
        
        template<typename F, typename... Args>
        auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
        
        void wait_for_all();
        size_t get_thread_count() const;
        
    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex queue_mutex_;
        std::condition_variable condition_;
        std::condition_variable finished_;
        std::atomic<bool> stop_;
        std::atomic<size_t> busy_threads_;
    };
    
    // Simple task scheduler
    class TaskScheduler {
    public:
        static TaskScheduler& instance();
        
        void schedule_task(std::function<void()> task, std::chrono::milliseconds delay);
        void schedule_repeating_task(std::function<void()> task, 
                                   std::chrono::milliseconds interval);
        
        void start();
        void stop();
        
    private:
        TaskScheduler() = default;
        // Implementation details...
    };
}

// Utility functions for circuit-specific operations
namespace circuit_utils {
    // Gate logic functions
    SignalValue evaluate_gate(GateType type, const std::vector<SignalValue>& inputs);
    
    // Truth table generation
    std::vector<std::vector<SignalValue>> generate_truth_table(uint32_t num_inputs);
    
    // Circuit analysis
    bool is_circuit_combinational(const Circuit& circuit);
    uint32_t calculate_circuit_depth(const Circuit& circuit);
    std::vector<GateId> get_critical_path(const Circuit& circuit);
    
    // Test case utilities
    std::vector<TestCase> generate_exhaustive_test_cases(uint32_t num_inputs, uint32_t num_outputs);
    std::vector<TestCase> generate_random_test_cases(uint32_t num_inputs, uint32_t num_outputs, 
                                                    uint32_t num_cases);
    
    // Performance benchmarking
    void benchmark_circuit_simulation(const Circuit& circuit, const std::vector<TestCase>& test_cases);
    void benchmark_genetic_algorithm(const EvolutionaryParams& params);
    
    // Format conversions
    std::string circuit_to_verilog(const Circuit& circuit);
    std::string circuit_to_dot(const Circuit& circuit);
    std::string genome_to_json(const Genome& genome);
    std::unique_ptr<Genome> json_to_genome(const std::string& json);
}

} // namespace circuit 