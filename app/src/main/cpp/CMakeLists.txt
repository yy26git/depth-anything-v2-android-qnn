cmake_minimum_required(VERSION 3.22.1)
project(depth_uint16 LANGUAGES CXX)
add_library(depth_uint16 SHARED depth_uint16.cpp)
target_compile_features(depth_uint16 PRIVATE cxx_std_17)
target_compile_options(depth_uint16 PRIVATE -Wall -Wextra -Werror)
target_include_directories(depth_uint16 PRIVATE ort_include)
# Link only; the identical pinned AAR owns packaging libonnxruntime.so.
target_link_directories(depth_uint16 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ort_link")
target_link_libraries(depth_uint16 PRIVATE onnxruntime log)
