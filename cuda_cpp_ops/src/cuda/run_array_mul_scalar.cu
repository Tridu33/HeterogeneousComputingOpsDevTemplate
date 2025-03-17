#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
__global__ void kernel4array_mul_scalar (double *vec, double scalar, int num_elements) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    vec[idx] = vec[idx] * scalar;
  }
}


void run_kernel4array_multiply_with_scalar(double *vec, double scalar, int num_elements) {
  dim3 dimBlock(256, 1, 1);
  dim3 dimGrid(ceil((double)num_elements / dimBlock.x));

  kernel4array_mul_scalar<<<dimGrid, dimBlock>>>(vec, scalar, num_elements); // 返回值保存在vec

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel4array_mul_scalar launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}
