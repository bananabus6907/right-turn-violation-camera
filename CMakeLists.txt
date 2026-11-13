# CMake 기본 설정
cmake_minimum_required(VERSION 3.16)

project(untitled LANGUAGES CUDA CXX)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# OpenCV 설정
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Python 설정
# find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
# include_directories(${Python3_INCLUDE_DIRS})

# LibTorch 설정
set(Torch_DIR /home/airobot/libtorch/share/cmake/Torch) # LibTorch가 설치된 경로
# set(Torch_DIR "/home/airobot/libtorch/share/cmake/Torch") # LibTorch가 설치된 경로
# set(CMAKE_PREFIX_PATH "/home/airobot/libtorch")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -D_GLIBCXX_USE_CXX11_ABI=1")


find_package(Torch REQUIRED)
if(Torch_FOUND)
    message(STATUS "Found Libtorch at ${Torch_DIR}")
    include_directories(${TORCH_INCLUDE_DIRS})  # LibTorch의 include 디렉토리 추가
else()
    message(FATAL_ERROR "Libtorch not found!")
endif()

# CUDA 및 cuDNN 설정
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.3")
set(CUDNN_INCLUDE_DIR "/usr/local/cuda-11.3/include")
set(CUDNN_LIBRARY "/usr/local/cuda-11.3/targets/x86_64-linux/lib/libcudnn.so")

find_package(CUDA 11.3 REQUIRED)
if(CUDA_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION}")
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${CUDNN_INCLUDE_DIR})
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
else()
    message(FATAL_ERROR "CUDA not found!")
endif()

# 실행 파일 생성
add_executable(untitled main.cpp)

# 라이브러리 연결
target_link_libraries(untitled
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}            # LibTorch 라이브러리 링크
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARY}
    # /usr/local/cuda-12.1/targets/x86_64-linux/lib/libnvrtc.so # libnvrtc.so 추가
    ${Python3_LIBRARIES}
    # stdc++fs                      # Filesystem support
)

# 디버그 빌드 설정
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Debug build: Enabling debugging symbols")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
    message(STATUS "Release build: Enabling optimizations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# 추가 설정
target_include_directories(untitled PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TORCH_INCLUDE_DIRS}         # LibTorch include 디렉토리 추가
)

# 빌드 출력 정보
set_property(TARGET untitled PROPERTY CXX_STANDARD 17)

# 출력 메시지
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Torch_DIR: ${CMAKE_PREFIX_PATH}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "CUDA Toolkit Directory: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "cuDNN Include Directory: ${CUDNN_INCLUDE_DIR}")
message(STATUS "Torch Include Directories: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
