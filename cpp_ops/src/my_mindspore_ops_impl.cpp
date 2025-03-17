#include "my_mindspore_ops_interface.hpp"
#include<iostream>
using namespace std;
int add_op(int i, int j){
    printf("i: %d, j: %d\n", i, j); //lldb,c, breakpoint set -f my_mindspore_ops_impl.cpp -l 5
    int ret = i + j;
    return ret;
}
//  strace -p 13251
// attach --pid 114514