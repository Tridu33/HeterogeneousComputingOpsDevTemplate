# 指定CMake编译最低要求版本
cmake_minimum_required(VERSION 2.8...3.5)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# include .h
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inc)
include_directories(${pybind11_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

find_package(CUDA) # 用find_package来导入CUDA包，然后就可以使用cuda_add_executable()或者cuda_add_library()来编译CUDA可执行文件或者库文件
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(CMAKE_LINK_DEF_FILE_FLAG "${CMAKE_LINK_DEF_FILE_FLAG} -L/usr/local/cuda-12/lib64") # 指定libcudaart.so之类的cuda动态链接库

# # pybind11 lib build
# pybind11_add_module(gpu_library ${CMAKE_CURRENT_SOURCE_DIR}/gpu_library.cu)

# # specify install directories
# install(TARGETS gpu_library LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

# ===========================================================================

# if(CUDA_ENABLE)
# enable_language(CUDA)
# endif()
# # cuda lib build
# add_library(cudaMatMul cudaMatMul.cu cudaMatMul.h)
# # compile cuda lib to c++ lib
# target_compile_features(cudaMatMul PUBLIC cxx_std_11)
# # c++ executable
# add_executable(main main.cpp)
# # link c++ lib
# target_link_libraries(main cudaMatMul)

# 遍历目录下全部函数实现的源文件
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/cpp MY_CPP_OPS_INCLUDE_LIST)

# c++ lib build
add_library(cpp_ops STATIC ${MY_CPP_OPS_INCLUDE_LIST})

# cuda lib build
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/cuda MY_CUDA_OPS_IMPLEMENT_LIST)
add_library(cuda_ops STATIC ${MY_CUDA_OPS_IMPLEMENT_LIST}) #等价写法: add_library(cuda_ops STATIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda/add.cu ${CMAKE_CURRENT_SOURCE_DIR}/cuda/run_array_mul_scalar.cu)

# set cuda properties
set_target_properties(cuda_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# c++ pybind11 lib build
pybind11_add_module(my_gpu_ops ${CMAKE_CURRENT_SOURCE_DIR}/my_gpu_ops_binder.cpp)

# link
link_directories(${CMAKE_CURRENT_BINARY_DIR})   # 暴露库路径
target_link_libraries(my_gpu_ops PRIVATE pybind11::module pybind11::embed cuda_ops cpp_ops)
set_target_properties(my_gpu_ops PROPERTIES    PREFIX "${PYTHON_MODULE_PREFIX}"
                                            SUFFIX "${PYTHON_MODULE_EXTENSION}")

# specify install directories
install(FILES ${PROJECT_SOURCE_DIR}/inc/my_gpu_ops.h DESTINATION ${CMAKE_INSTALL_PREFIX}/inc)
install(TARGETS cuda_ops LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS cpp_ops LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS my_gpu_ops LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
