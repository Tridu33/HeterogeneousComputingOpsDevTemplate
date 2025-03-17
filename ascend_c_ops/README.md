小白扫盲推荐阅读[昇腾Ascend TIK自定义算子开发教程（概念版）](https://blog.csdn.net/m0_37605642/article/details/132780001)

> TBE框架给用户提供自定义算子，包括TBE DSL、TBE TIK、AICPU三种开发方式，TIK用Python写算子，TIK2用c++写算子。TBE是上一代的算子开发语言了，华为TBE和AKG都基于TVM但是对动态规模的支持不是很好。目前TBE不怎么演进了，都逐步走向**AscendC**(旧名TIK C++/TIK2)。

## 环境准备

**Toolkit** 开发套件 本质上包含了离线推理引擎(NNRT)和实用工具(Toolbox),所以不管是运行环境还是开发环境，只要安装了Toolkit就行。

```bash
install_path=/usr/local/Ascend/ascend-toolkit/latest 
source ${install_path}/bin/setenv.bash
export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub 
```

使用CANN运行用户编译、运行时，需要以CANN运行用户登录环境，执行 `source ${install_path}/set_env.sh`命令设置环境变量，其中 `${install_path}`为CANN软件的安装目录
`export ASCEND_INSTALL_PATH=/usr/local/Ascend可以设置环境变量以备后续使用。`

- 运行环境安装nnrt包，则开发过程中引用对应AscendCL目录。
  头文件路径：CANN软件安装后文件存储路径 `${ASCEND_INSTALL_PATH}/nnrt/latest/include/acl`
  库文件路径：CANN软件安装后文件存储路径 `${ASCEND_INSTALL_PATH}/nnrt/latest/lib64`
- 运行环境安装nnae包，则开发过程中引用对应AscendCL目录。
  头文件路径：CANN软件安装后文件存储路径 `${ASCEND_INSTALL_PATH}/nnae/latest/include/acl`
  库文件路径：CANN软件安装后文件存储路径 `${ASCEND_INSTALL_PATH}/nnae/latest/lib64`

Ascend CL（TIK2 C++）算子可用CPU模式或NPU模式执行

- CPU模式： 算子功能调试用，可以模拟在NPU上的计算行为，不需要依赖昇腾设备

```cpp
#include "tikicpulib.h"
#define_aicore_
```

- NPU模式： 算子功能/性能调试用，可以使用NPU的强大算力进行运算加速

```cpp
#include "acl/acl.h"
#define_aicore [aicore]
```

（可选）通过环境变量ASCEND_CACHE_PATH、ASCEND_WORK_PATH设置AscendCL应用编译运行过程中产生的文件的落盘路径，涉及ATC模型转换、AscendCL应用编译配置、AOE模型智能调优、性能数据采集、日志采集等功能，落盘文件包括算子编译缓存文件、知识库文件、调优结果文件、性能数据文件、日志文件等。

```bash
export ASCEND_CACHE_PATH=/repo/task001/cache
export ASCEND_WORK_PATH=/repo/task001/172.16.1.12_01_03
```

## Ascend C demos

动态图使用的aclnn算子清单可以查询官网，比如[aclnn矩阵乘法算子](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/aolapi/context/aclnnMatmul.md)等。

为了讲解如何开发Ascend CL的算子，建议先了解一下面几个案例：

- [helloworld初识开发流程](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/0_helloworld)，
- [Matmul算子的核函数直调方法](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/11_matmul_kernellaunch)
- [基于MatmulCustom算子工程，介绍了单算子工程、单算子调用](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/10_matmul_frameworklaunch)
- 开发者完成kernel侧算子实现和host侧tiling实现后，即可通过AscendCL运行时接口，完成[算子kernel直调](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0056.html)。该方式下tiling开发不受CANN框架的限制，简单直接，多用于算子功能的快速验证。
- 对于KernelLaunch开放式算子编程的方式，通过[第三方框架Pybind11调用AscendC算子](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0057.html)，可以实现Python调用算子kernel程序。

**注意**：生产环境可以考虑断点保存MindSpore的算子输入数据 `numpy_input_data = mindsporeTensor.asnumpy()`，根据[numpy输入数据导出bin文件用于AscendC算子开发的案例](https://gitee.com/ascend/samples/blob/master/cplusplus/level1_single_api/4_op_dev/6_ascendc_custom_op/kernel_invocation/MatMul/matmul_custom.py)的做法导出Python中的输入数据 `input_data.bin`，然后导入到Ascend C中做算子开发。类比CPU上传统的，GPU上编程模型SIMT对应SPMD执行模型可以用来实现各种高性能并行计算，AscendC语言提供“释放NPU算力去承接诸如图片渲染等大规模并行计算任务”的一种方案，比如[DVPP](https://bbs.huaweicloud.com/blogs/394593?utm_source=zhihu&utm_medium=bbs-ex&utm_campaign=other&utm_content=content))。如果未来Ascend C对接上Triton DSL，就可能只需要写一份python代码自定义算子，然后GPU/NPU上分别编译为CUDA/或scendC，甚至跳过这两个接口语言，直接编译到PTX之类的二进制bin直接执行。

- `kernel_invocation`目录是[Ascend C保姆级教程](https://www.hiascend.com/forum/thread-0239124507219723020-1-1.html)，从一个简单的add实例出发，带你体验Ascend C算子开发的基本流程。相对路径引用了 `kernel_template`目录下一些公共函数和公共文件，`ll`命令可以参考 https://gitee.com/ascend/samples/tree/v0.2-8.0.0.beta1/cplusplus/level1_single_api/4_op_dev/6_ascendc_custom_op/kernel_invocation 多个开发案例启动命令的方法。因为AscendC主力支持的SOC_VERSION列表是 [Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4]，如果没有对应设备就要考虑 `cpu`模式模拟执行。
- `ascend_c_ops`目录[基于AscendC的add算子](https://gitee.com/ascend/samples/tree/v0.2-8.0.0.beta1/operator/ascendc/0_introduction/3_add_kernellaunch/CppExtensions#/ascend/samples/blob/v0.2-8.0.0.beta1/operator/ascendc/0_introduction/3_add_kernellaunch/CppExtensions/./add_custom.cpp)，介绍如何对标CUDA为NPU自定义算子，并通过Pybind11暴露给Python侧调用，提供一个最小可用开发原型。
- `ms_ops_custom_ascendc`目录[MindSpore官网自定义AscendC的NPU算子](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/custom_program/operation/op_custom_ascendc.html)考虑框架本身和多后端兼容。
