cmake_minimum_required(VERSION 3.4...3.18)
project(my_mindspore_ops)

# ------- DEBUG添加这两行 -------
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

add_library(my_mindspore_ops_impl STATIC src/my_mindspore_ops_impl.cpp)
set_target_properties(my_mindspore_ops_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(../third_party/pybind11 ../third_party/pybind11)
pybind11_add_module(my_mindspore_ops src/my_mindspore_ops_binder.cpp) # 最后Python脚本中 import my_mindspore_ops 通过查找my_mindspore_ops_interface.hpp接口
target_link_libraries(my_mindspore_ops PRIVATE my_mindspore_ops_impl) # 接口hpp 绑定 实现cpp
