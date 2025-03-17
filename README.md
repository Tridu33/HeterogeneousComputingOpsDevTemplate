# Operator development template for Pybind11 binding

[This project](https://github.com/Tridu33/OperatorsDevTemplate/tree/main) uses the most basic TensorAdd as an example to introduce the operator development call process of "Python is not skinned, C++ is winged". More operators are considered first and then rewrite:

- CPU,[More CPU quantization operator reference llamafile](https://github.com/Mozilla-Ocho/llamafile/tree/main/llama.cpp), you need to know the usage of the AVX instruction set of x86 and the usage of the NEON instruction set of arm64;
- CUDA.cu for GPU,[More GPU inference operator reference CUDA official samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples),[Teacher Fan Zheyong's book "CUDA-Programming Programming"](https://github.com/brucefan1983/CUDA-Programming) and [CUDA_kernel_Samples](https://github.com/Tongkaio/CUDA_Kernel_Samples) similar case sets, you need to understand parallel development of CUDA and pyCUDA;-
- Ascend NPU,[More examples of NPU inference operator reference official solutions](https://github.com/Ascend/samples/tree/master/cplusplus/level1_single_api/4_op_dev/1_custom_op) and [Old Tan taking off at Station B](https://space.bilibili.com/668461244?spm_id_from=333.337.0.0) for information such as TBE, pyACL, OMl quantitative inference operator library, etc., you need to know the pre-knowledge of Ascend systems, and find it on demand or rewritten it yourself.
- Python operators: Operators written in native Python do not require pybind. [Triton official tutorials](https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py) and [Awesome-Triton-Kernels](https://github.com/zinccat/Awesome-Triton-Kernels).

JAX is an experimental field of AI frameworks for autograd+XLA in pure function differential programming. It exists analogously with PyTorch, MindSpore, Tensorflow, etc. The FP programming philosophy lies in the flexibility of composability, such as [cuda+cpp write operator pybind11 encapsulates an operator to call](https://jax.ac.cn/en/latest/Custom_Operation_for_GPUs.html).

In fact, the Python AI framework and the underlying operator are decoupled:

1. CopyIn task: input H2D pointer is passed to the underlying heterogeneous operator;
2. Compute tasks: Automatic (the mainstream practice of inference engines such as llama.cpp is to automatically select the backend based on availability) or manually (ktransformers use yaml to manually specify the specific structure of MoE to heterogeneous devices) to assign tasks in the dispatch calculation diagram to the specific operator of heterogeneous computing structure.so;
3. CopyOut: Task. Finally, return the calculation result to the Python caller through D2H;

This project first introduces how pybind11 calls cpp and debug it; then introduces how cuda and cpp operators are bound and used on GPU machines; then introduces the introduction to the development of AscendNPU operators. for example [MindSpore first registers an operator interface, and can use similar methods to dispatch to CPU, GPU, and NPU multi-terminal implementation.](https://github.com/openmlsys/openmlsys-zh/blob/main/chapter_programming_interface/c_python_interaction.md).
