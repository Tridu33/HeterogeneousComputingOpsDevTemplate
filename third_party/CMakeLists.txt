# 指定CMake编译最低要求版本
cmake_minimum_required(VERSION 2.8...3.5)

# 设置c++标准
set(PYBIND11_CPP_STANDARD -std=c++14)

# 编译构建子目录
add_subdirectory(pybind11)