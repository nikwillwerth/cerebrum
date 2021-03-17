cmake_minimum_required(VERSION 3.17)
project(cerebrum)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(cerebrum main.cpp cerebrum/layers/Layer.cpp cerebrum/layers/Layer.h cerebrum/layers/Dense.cpp cerebrum/layers/Dense.h cerebrum/layers/Input.cpp cerebrum/layers/Input.h cerebrum/utils/TensorOps.cpp cerebrum/utils/TensorOps.h cerebrum/layers/Softmax.cpp cerebrum/layers/Softmax.h cerebrum/layers/ReLU.cpp cerebrum/layers/ReLU.h)
target_link_libraries(cerebrum Eigen3::Eigen)