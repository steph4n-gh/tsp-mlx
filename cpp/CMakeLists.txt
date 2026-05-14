cmake_minimum_required(VERSION 3.24)
project(tsp_mlx_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point to MLX
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../venv/lib/python3.14/site-packages/mlx")
find_package(mlx REQUIRED)

# Include directories
include_directories(include ${MLX_INCLUDE_DIRS})

# Add Rust library
add_library(tau_gate STATIC IMPORTED)
set_target_properties(tau_gate PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/../../supplychain/target/release/libtau_gate.a"
)

# Core library
add_library(tsp_core STATIC
    src/sparse_cache.cpp
    src/position_tracker.cpp
    src/attention_hook.cpp
    src/compression.cpp
    src/consolidation.cpp
)

target_link_libraries(tsp_core PRIVATE mlx tau_gate)

# Main executable
add_executable(tsp_inference src/main.cpp)
target_link_libraries(tsp_inference PRIVATE tsp_core mlx)

# Demo executable
add_executable(tsp_demo src/demo.cpp)
target_link_libraries(tsp_demo PRIVATE tsp_core mlx)
