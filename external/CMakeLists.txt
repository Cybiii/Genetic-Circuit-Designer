# External Dependencies CMakeLists.txt

# ImGui setup
set(IMGUI_SOURCES
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp
)

add_library(imgui STATIC ${IMGUI_SOURCES})

target_include_directories(imgui PUBLIC
    imgui/
    imgui/backends/
)

target_link_libraries(imgui PUBLIC
    ${OPENGL_LIBRARIES}
    glfw
)

# GLAD setup (OpenGL loader)
set(GLAD_SOURCES
    glad/src/glad.c
)

add_library(glad STATIC ${GLAD_SOURCES})

target_include_directories(glad PUBLIC
    glad/include/
)

target_link_libraries(glad PUBLIC
    ${OPENGL_LIBRARIES}
    ${CMAKE_DL_LIBS}
)

# nlohmann/json is header-only, so we just need to set up the include path
add_library(nlohmann_json INTERFACE)

target_include_directories(nlohmann_json INTERFACE
    nlohmann/
)

# Export targets to parent scope
set(EXTERNAL_LIBRARIES
    imgui
    glad  
    nlohmann_json
    PARENT_SCOPE
)

# Download and setup external dependencies if they don't exist
include(FetchContent)

# ImGui
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui.h)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.89.2
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/imgui
    )
    FetchContent_MakeAvailable(imgui)
endif()

# GLAD
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/glad/include/glad/glad.h)
    FetchContent_Declare(
        glad
        GIT_REPOSITORY https://github.com/Dav1dde/glad.git
        GIT_TAG v0.1.36
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/glad
    )
    FetchContent_MakeAvailable(glad)
endif()

# nlohmann/json
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann/json.hpp)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Alternative: Download single header file for nlohmann/json
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann/json.hpp)
    file(DOWNLOAD 
        https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann/json.hpp
        SHOW_PROGRESS
    )
endif()

# Create directories if they don't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/glad)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann) 