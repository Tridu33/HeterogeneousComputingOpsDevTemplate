#!/bin/bash
clear;clear
# 清除之前遗留的文件
rm -rf *.vcd *.dump *.log *.bin *.o *.so *pu build output/*.bin input/*.bin
# 不需要TIK打印出内存信息
export PRINT_TIK_MEM_ACCESS=TRUE

# 获取当前的目录
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
); cd $CURRENT_DIR

declare -A VersionMap
VersionMap["ascend910"]="Ascend910A"
VersionMap["ascend310p"]="Ascend310P1"
VersionMap["ascend910B1"]="Ascend910B1"

# 指向昇腾软件包安装地址，导出环境变量
if [ ! $ASCEND_HOME_DIR ]; then
    export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
fi
source $ASCEND_HOME_DIR/bin/setenv.bash


# 指定当前sample的算子文件名
FILE_NAME=$1

# 指定芯片版本: ascend910, ascend310p
SOC_VERSION=$2
if [ ${SOC_VERSION}"x" = "x" ]; then
    echo "ERROR: SOC_VERSION is not specified! please specify ascend910, ascend310p or ascend910B1!"
    exit -1
fi

# 指定运行的核: AiCore, VectorCore
CORE_TYPE=$3
if [ ${CORE_TYPE}"x" = "x" ]; then
    echo "WARNING: CORE_TYPE is not specified, using AiCore as default."
    CORE_TYPE=AiCore
fi

# 指定运行模式: cpu, npu
RUN_MODE=$4
if [ ${RUN_MODE}"x" = "x" ]; then
    echo "WARNING: RUN_MODE is not specified, using cpu as default."
    RUN_MODE=cpu
fi

# 生成计算输入数据和对比用的真值数据
python3 $FILE_NAME.py

function compile_and_execute() {
    # 使用cmake编译cpu侧或者npu侧算子, SIMULATOR or ONBOARD
    mkdir -p build; cd build;       \
    cmake ..                        \
        -Dsmoke_testcase=$1         \
        -DASCEND_PRODUCT_TYPE=$2    \
        -DASCEND_CORE_TYPE=$3       \
        -DASCEND_RUN_MODE="ONBOARD" \
        -DASCEND_INSTALL_PATH=$ASCEND_HOME_DIR
    cmake --build . --target ${1}_${4}
    # cmake --build . --target ascendc_kernels && cmake --install . && cmake --build . --target ${1}_lib_${4}
    cd -

    if [ $? -ne 0 ]; then
        echo "ERROR: compile op on failed!"
        return 1
    fi
    echo "INFO: compile op on ${RUN_MODE} succeed!"

    # 执行生成的可执行文件
    (export LD_LIBRARY_PATH=`pwd`:$ASCEND_HOME_DIR/tools/simulator/${VersionMap[$SOC_VERSION]}/lib:$LD_LIBRARY_PATH && ./${1}_${4})
    if [ $? -ne 0 ]; then
        echo "ERROR: execute op on ${RUN_MODE} failed!"
        return 1
    fi
    echo "INFO: execute op on ${RUN_MODE} succeed!"
}
compile_and_execute $FILE_NAME $SOC_VERSION $CORE_TYPE $RUN_MODE

# 验证计算结果
echo "md5sum: ";md5sum output/*.bin
