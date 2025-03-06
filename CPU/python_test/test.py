#!python
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))  # sys.path.append(os.path.join(os.path.dirname(__file__), '../package_3'))
import my_mindspore_ops

a = 1
b = my_mindspore_ops.add_op(1, 2)
c = 2
print(b)
d = my_mindspore_ops.subtract_op(514,114)
print('514-114=',d)
