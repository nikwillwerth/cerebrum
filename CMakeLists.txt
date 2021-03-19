cmake_minimum_required(VERSION 3.17)
project(cerebrum)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native -O3")

find_package(CURL REQUIRED)

include_directories(${CURL_INCLUDE_DIR})
include_directories(3rd_party/Eigen)
include_directories(3rd_party/include)
include_directories(3rd_party)

add_executable(cerebrum main.cpp cerebrum/layers/Layer.cpp cerebrum/layers/Layer.h cerebrum/layers/Dense.cpp cerebrum/layers/Dense.h cerebrum/layers/Input.cpp cerebrum/layers/Input.h cerebrum/utils/TensorOps.cpp cerebrum/utils/TensorOps.h cerebrum/layers/Softmax.cpp cerebrum/layers/Softmax.h cerebrum/layers/ReLU.cpp cerebrum/layers/ReLU.h cerebrum/layers/Conv2D.cpp cerebrum/layers/Conv2D.h cerebrum/data/Mnist.cpp cerebrum/data/Mnist.h cerebrum/Model.cpp cerebrum/Model.h cerebrum/optimizers/Optimizer.cpp cerebrum/optimizers/Optimizer.h cerebrum/optimizers/SGD.cpp cerebrum/optimizers/SGD.h cerebrum/utils/NetIO.cpp cerebrum/utils/NetIO.h)
target_link_libraries(cerebrum ${CURL_LIBRARIES} -lz)