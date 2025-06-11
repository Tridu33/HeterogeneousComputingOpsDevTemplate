/**
 * @file matmul_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

template <typename aType, typename bType, typename cType, typename biasType> class MatmulKernel {
public:
    __aicore__ inline MatmulKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling);
    template <bool setTmpSpace = false> __aicore__ inline void Process(TPipe *pipe);

    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, aType>, MatmulType<TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<TPosition::GM, CubeFormat::ND, cType>, MatmulType<TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    GlobalTensor<aType> aGlobal;
    GlobalTensor<bType> bGlobal;
    GlobalTensor<cType> cGlobal;
    GlobalTensor<biasType> biasGlobal;
    TCubeTiling tiling;
};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                                         GM_ADDR workspace, const TCubeTiling &tiling)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);

    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetC = 0;
    int32_t offsetBias = 0;
    CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename aType, typename bType, typename cType, typename biasType>
template <bool setTmpSpace>
__aicore__ inline void MatmulKernel<aType, bType, cType, biasType>::Process(TPipe *pipe)
{
    if constexpr (setTmpSpace) {
        TBuf<> tmpMMFormatUb;
        LocalTensor<uint8_t> mmformatUb;
        pipe->InitBuffer(tmpMMFormatUb, TOTAL_VEC_LOCAL_SIZE);
        mmformatUb = tmpMMFormatUb.Get<uint8_t>(TOTAL_VEC_LOCAL_SIZE);
        matmulObj.SetLocalWorkspace(mmformatUb);
    }

    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    matmulObj.IterateAll(cGlobal);
    matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA,
                                                        int32_t &offsetB, int32_t &offsetC, int32_t &offsetBias)
{
    auto mSingleBlocks = Ceil(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulKernel<half, half, float, float> matmulKernel;
    TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tilingData.cubeTilingData);
    matmulKernel.Init(a, b, bias, c, workspace, tilingData.cubeTilingData);
    if (TILING_KEY_IS(1)) {
        matmulKernel.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        matmulKernel.Process<true>(&pipe);
    }
}