#!/bin/bash
script_dir=$(dirname "$0")
export ASCEND_CUSTOM_OPP_PATH="$script_dir/ascendcSample4AddCustom/myinstallpath/vendors/customize/":${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=$script_dir/ascendcSample4AddCustom/myinstallpath/vendors/aclnnAddCustom/op_api/lib/:${LD_LIBRARY_PATH}
export PYTHONPATH=$PYTHONPATH:/root/mindspore

python ./tests/test_add_custom_aclnn_can_be_disapear.py  
python ./tests/test_customize_aclnnAddCustom.py


