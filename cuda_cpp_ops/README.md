```bash
mkdir build && cd build
rm -rf ./* && cmake .. && make -j192 && python ../test/test_add.py 
python ../test/test_multiply_with_scalar.py 
cd .. && nvcc -I ./inc ./main4cpp_call_cuda_kernel.cpp ./src/cuda/run_kernel4elementwise_add_float4.cu -o ./manual_main4test_elementwise_add_float4 && ./manual_main4test_elementwise_add_float4

```
