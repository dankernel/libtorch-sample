cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

# Add torch and OpenCV lib
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Add OpenCV include dirs
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(example-app main.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
