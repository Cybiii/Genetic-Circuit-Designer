# External dependencies CMakeLists.txt

# Only include external dependencies if visualization is enabled
if(ENABLE_VISUALIZATION)
    # Add OpenGL loader (GLAD) if available
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/glad/CMakeLists.txt)
        add_subdirectory(glad)
    endif()
    
    # Add ImGui if available
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
        # ImGui setup would go here
    endif()
endif()

# nlohmann/json is header-only, so we just need to make sure it's available
# No additional build steps needed for nlohmann/json 