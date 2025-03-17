#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(a) *(float4 *)(&(a))

#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

// 核函数，纯纯放进去GPU执行的逻辑
__global__ void kernel4elementwise_add_float4(float* a, float* b, float* c, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= N) return;
    
    float4 tmp_a = FLOAT4(a[idx]);
    float4 tmp_b = FLOAT4(b[idx]);
    float4 tmp_c;
    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;
    FLOAT4(c[idx]) = tmp_c;
}

// 1. host数据传进去核函数
// 2. <<<调用核函数计算>>>
// 3. GPU数据指针c_device，把计算结果返回host数据指针c_host
void run_kernel4elementwise_add_float4(float* a_h,
                                       float* b_h,
                                       float* c_h,
                                       int N) {
  float* a_d = nullptr;
  float* b_d = nullptr;
  float* c_d = nullptr;
  cudaCheck(cudaMalloc((void**)&a_d, N * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&b_d, N * sizeof(float)));
  cudaCheck(cudaMalloc((void**)&c_d, N * sizeof(float)));
  cudaCheck(cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(b_d, b_h, N * sizeof(float), cudaMemcpyHostToDevice));
  int block_size = 1024;
  int grid_size = CEIL(CEIL(N, 4), 1024);
  kernel4elementwise_add_float4<<<grid_size, block_size>>>(a_d, b_d, c_d, N);
  // D2H 返回值保存在c_h
  cudaCheck(cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));
}
