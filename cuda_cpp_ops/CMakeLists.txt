# 指定CMake编译最低要求版本
cmake_minimum_required(VERSION 2.8...3.5)

# 给项目命名,解决CMP0048
if(CMAKE_VERSION VERSION_LESS 3.0)
  set(PROJECT_VERSION 0)
else()
  cmake_policy(SET CMP0048 NEW)
endif()

# PROJECT_NAME
project(pybind11_cuda LANGUAGES CXX CUDA)

# c++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g ")

# cuda flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC -g")

# 寻找CUDA工具包
# find_package(CUDAToolkit REQUIRED)   # 不同cmake版本
find_package(CUDA REQUIRED)

# 支持cuda架构sm_75;sm_80
set(CMAKE_CUDA_ARCHITECTURES 75;80)

# 指定安装路径
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/out CACHE INTERNAL "install directories")

# 编译构建子目录
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party ${CMAKE_CURRENT_SOURCE_DIR}/../third_party) 
add_subdirectory(src)




