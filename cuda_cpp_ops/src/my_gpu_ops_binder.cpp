#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

#include "my_gpu_ops.h"
#include "run_array_mul_scalar.hpp"  // 只需 函数声明，编译的时候链接obj文件中寻找 函数实现

namespace py = pybind11;

void add_wrapper(py::array_t<int> a, py::array_t<int> b) {
    auto buf_a = a.request(), buf_b = b.request();
    int *ptr_a = static_cast<int *>(buf_a.ptr);
    int *ptr_b = static_cast<int *>(buf_b.ptr);

    // Call the CUDA function through the header file
    int size = buf_a.size;
    int *c = new int[size];
    add(c, ptr_a, ptr_b, size);
    printf("cpp Call: c %d a %d b %d size %d\n", *c, *ptr_a, *ptr_b, size);

    // Create a new numpy array to return the result
    py::array_t<int> result = py::array_t<int>(size);
    auto buf_result = result.request();
    int *ptr_result = static_cast<int *>(buf_result.ptr);

    // Copy the result from the CUDA function to the numpy array
    for (int i = 0; i < size; i++) {
        ptr_result[i] = c[i];
        printf("cpp: c %d\n", ptr_result[i]);
    }

    delete[] c;
}


// pybind11.h函数封装：传参用上pybind11定义的数据结构，保证python调用cpp的时候类型转换不出错
void array_multiply_with_scalar(pybind11::array_t<double> vec, double scalar) {
  int size = 10;
  double* gpu_ptr;
  cudaError_t error = cudaMalloc(&gpu_ptr, size * sizeof(double));

  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
  auto ha = vec.request();

  if (ha.ndim != 1) {
    std::stringstream strstr;
    strstr << "ha.ndim != 1" << std::endl;
    strstr << "ha.ndim: " << ha.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  double* ptr = reinterpret_cast<double*>(ha.ptr);
  error =
      cudaMemcpy(gpu_ptr, ptr, size * sizeof(double), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  run_kernel4array_multiply_with_scalar(gpu_ptr, scalar, size);

  error =
      cudaMemcpy(ptr, gpu_ptr, size * sizeof(double), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaFree(gpu_ptr);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

PYBIND11_MODULE(my_gpu_ops, m) {
  m.def("add", &add_wrapper, "Add two arrays using CUDA");
  m.def("add_cpp", &add_cpp, "Add two numbers using cpp");
  m.def("array_multiply_with_scalar", array_multiply_with_scalar);
}
