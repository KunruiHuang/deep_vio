cmake_minimum_required(VERSION 3.10)
project(TorchDemo)

set(CMAKE_CXX_STANDARD 14)
set(Torch_DIR /home/hkr/mnt/lib/libtorch-cxx11-abi-shared-with-deps-1.8.2+cu102/libtorch/share/cmake/Torch)
set(OpenCV_DIR /home/hkr/mnt/opencv-3.4.4/build)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(TorchDemo src/main.cpp)
target_link_libraries(TorchDemo ${OpenCV_LIBS} ${TORCH_LIBRARIES} )