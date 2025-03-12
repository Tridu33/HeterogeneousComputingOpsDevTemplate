```bash

# 1.先切换回项目的根目录
# 2.
mkdir build &&  cd build
# 3.-DPYTHON_EXECUTABLE是python的路径
# cmake .. -DPYTHON_EXECUTABLE=/Users/username/miniconda3/bin/python
cmake .. -DPYTHON_EXECUTABLE=$(which python)
# 4.
cd ..
make all
file my_mindspore_ops.cpython-312-x86_64-linux-gnu.so
python setup.py build_ext --inplace
# 然后就生成了一个.so文件
python setup.py bdist_wheel
# 创建whl文件

```
