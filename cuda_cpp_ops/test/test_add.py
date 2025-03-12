import sys
import os

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pwd+'/../build/src')

import numpy as np
import my_gpu_ops

def main():
    # 创建两个NumPy数组
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6], dtype=np.int32)

    # 调用C++扩展模块中的add_wrapper函数
    result = my_gpu_ops.add(a, b)

    # 打印计算结果
    print(f"Cuda Result: {a} + {b} = {result}")
    
    # 调用C++扩展模块中的add_cpp函数
    a = 11
    b = 17
    result = my_gpu_ops.add_cpp(11, 17)
    
    # 打印计算结果
    print(f"Cpp Result: {a} + {b} = {result}")

if __name__ == "__main__":
    main()
