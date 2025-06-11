/**
 * @file matmul_leakyrelu_custom.cpp
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

template <typename aType, typename bType, typename cType, typename biasType> class MatmulLeakyKernel {
public:
    __aicore__ inline MatmulLeakyKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling, float alpha, TPipe *pipe);
    template <bool setTmpSpace = false> __aicore__ inline void Process(TPipe *pipe);

    __aicore__ inline void MatmulCompute();
    __aicore__ inline void LeakyReluCompute();
    __aicore__ inline void CopyOut(uint32_t count);
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, aType>, MatmulType<TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<TPosition::VECIN, CubeFormat::ND, cType>, MatmulType<TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    GlobalTensor<aType> aGlobal;
    GlobalTensor<bType> bGlobal;
    GlobalTensor<cType> cGlobal;
    GlobalTensor<biasType> biasGlobal;
    LocalTensor<cType> reluOutLocal;
    float alpha;
    TCubeTiling tiling;
    TQue<QuePosition::VECOUT, 1> reluOutQueue_;
};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                                       const TCubeTiling &tiling, float alpha, TPipe *pipe)
{
    this->tiling = tiling;
    this->alpha = alpha;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    int offsetBias = 0;
    CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
    pipe->InitBuffer(reluOutQueue_, 1, tiling.baseM * tiling.baseN * sizeof(cType));
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename aType, typename bType, typename cType, typename biasType>
template <bool setTmpSpace>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Process(TPipe *pipe)
{
    uint32_t computeRound = 0;
    if constexpr (setTmpSpace) {
        TBuf<> tmpMMFormatUb;
        LocalTensor<uint8_t> mmformatUb;
        pipe->InitBuffer(tmpMMFormatUb, tiling.baseM * tiling.baseN * sizeof(cType));
        mmformatUb = tmpMMFormatUb.Get<uint8_t>(tiling.baseM * tiling.baseN * sizeof(cType));
        matmulObj.SetLocalWorkspace(mmformatUb);
    }
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    while (matmulObj.template Iterate<true>()) {
        MatmulCompute();
        LeakyReluCompute();
        CopyOut(computeRound);
        computeRound++;
    }
    matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::MatmulCompute()
{
    reluOutLocal = reluOutQueue_.AllocTensor<cType>();
    matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::LeakyReluCompute()
{
    LeakyRelu(reluOutLocal, reluOutLocal, (cType)alpha, tiling.baseM * tiling.baseN);
    reluOutQueue_.EnQue(reluOutLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count)
{
    reluOutQueue_.DeQue<cType>();
    const uint32_t roundM = tiling.singleCoreM / tiling.baseM;
    const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
    uint32_t startOffset = (count % roundM * tiling.baseM * tiling.N + count / roundM * tiling.baseN);
    DataCopyParams copyParam = {(uint16_t)tiling.baseM, (uint16_t)(tiling.baseN * sizeof(cType) / DEFAULT_C0_SIZE), 0,
                                (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / DEFAULT_C0_SIZE)};
    DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
    reluOutQueue_.FreeTensor(reluOutLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                             int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                                             int32_t &offsetBias)
{
    auto mSingleBlocks = Ceil(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void matmul_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
    TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &tilingData.cubeTilingData);
    matmulLeakyKernel.Init(a, b, bias, c, workspace, tilingData.cubeTilingData, tilingData.alpha, &pipe);
    if (TILING_KEY_IS(1)) {
        matmulLeakyKernel.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        matmulLeakyKernel.Process<true>(&pipe);
    }
}