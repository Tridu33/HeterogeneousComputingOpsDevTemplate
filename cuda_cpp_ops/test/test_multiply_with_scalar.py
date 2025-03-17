#!python
import sys
import os

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pwd+'/../build/src')

import numpy as np
import my_gpu_ops

import numpy
vec = numpy.linspace(0,1,10)
print("before: ", vec)
my_gpu_ops.array_multiply_with_scalar(vec, 10)
print("after: ", vec)
