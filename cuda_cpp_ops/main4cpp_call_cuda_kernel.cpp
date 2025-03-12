

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "inc/run_kernel4elementwise_add_float4.hpp"

// this demo is modified from https://github.com/Tongkaio/CUDA_Kernel_Samples/blob/master/elementwise/add.cu (all in one)
int main() {
  // cpp直接引用run_kernel4elementwise_add_float4，和Pybind11的区别在于”pybind11命令空间内类型“的类型转换和函数函数封装
  // nvcc -I ./inc ./main4cpp_call_cuda_kernel.cpp ./src/cuda/run_kernel4elementwise_add_float4.cu -o ./manual_main4test_elementwise_add_float4 && ./manual_main4test_elementwise_add_float4
  // 类似于run_kernel4elementwise_add_float4的定义，Pybind11中只需要定义一个类似的函数，并把参数(a_h, b_h, c_h, N)类型转为pybind11命令空间的函数，即可python调用并获得结果
  constexpr int N = 7;
  float* a_h = (float*)malloc(N * sizeof(float));
  float* b_h = (float*)malloc(N * sizeof(float));
  float* c_h = (float*)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    a_h[i] = i;
    b_h[i] = N - 1 - i;
    }

    // 三步走，返回值保存在c_h
    run_kernel4elementwise_add_float4(a_h, b_h, c_h, N);

    printf("a_h:\n");
    for (int i = 0; i < N; i++ ) {
        if (i == N-1) printf("%f\n", a_h[i]);
        else printf("%f ", a_h[i]);
    }
    printf("b_h:\n");
    for (int i = 0; i < N; i++ ) {
        if (i == N-1) printf("%f\n", b_h[i]);
        else printf("%f ", b_h[i]);
    }
    printf("c_h = a_h+b_h:\n");
    for (int i = 0; i < N; i++ ) {
        if (i == N-1) printf("%f\n", c_h[i]);
        else printf("%f ", c_h[i]);
    }
    return 0;
}