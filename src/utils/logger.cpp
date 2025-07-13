#include "circuit/utils/utils.h"
#include <iostream>

namespace circuit {

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < min_level_) return;
    
    if (console_output_) {
        std::cout << level_to_string(level) << ": " << message << std::endl;
    }
}

void Logger::set_output_file(const std::string& filename) {
    // Stub implementation
}

std::string Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default: return "UNKNOWN";
    }
}

std::string Logger::get_timestamp() {
    return "[timestamp]";
}

} // namespace circuit 