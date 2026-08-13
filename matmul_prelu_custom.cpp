/**
 * @file matmul_prelu_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    // 唯一修改：uint32 -> uint32_t
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
}

template <typename aType, typename bType, typename cType, typename biasType> class MatmulPreluKernel {
public:
    __aicore__ inline MatmulPreluKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, cType alpha, GM_ADDR workspace,
                                const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void MatmulCompute();
    __aicore__ inline void PreluCompute();
    __aicore__ inline void CopyOut(uint32_t count);
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;
    AscendC::LocalTensor<cType> reluOutLocal;
    AscendC::LocalTensor<cType> tmpLocal;
    TCubeTiling tiling;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> reluOutQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    cType alpha;

    // 新增：标记当前 block 是否是合法的 tiling block
    bool isValidBlock = true;
};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulPreluKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias,
                                                                               GM_ADDR c, cType alpha,
                                                                               GM_ADDR workspace,
                                                                               const TCubeTiling &tiling,
                                                                               AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    this->alpha = alpha;

    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetC = 0;
    int32_t offsetBias = 0;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias);

    // 非法 block 仍然完整走流程，但只给安全长度，避免越界访问
    uint32_t aLen = isValidBlock ? (tiling.singleCoreM * tiling.Ka) : 1;
    uint32_t bLen = isValidBlock ? (tiling.Kb * tiling.singleCoreN) : 1;
    uint32_t cLen = isValidBlock ? (tiling.singleCoreM * tiling.N) : 1;
    uint32_t biasLen = isValidBlock ? tiling.singleCoreN : 1;

    aGlobal.SetGlobalBuffer((__gm__ aType *)a + offsetA, aLen);
    bGlobal.SetGlobalBuffer((__gm__ bType *)b + offsetB, bLen);
    cGlobal.SetGlobalBuffer((__gm__ cType *)c + offsetC, cLen);
    biasGlobal.SetGlobalBuffer((__gm__ biasType *)bias + offsetBias, biasLen);

    pipe->InitBuffer(reluOutQueue, 1, tiling.baseM * tiling.baseN * sizeof(cType));
    pipe->InitBuffer(tmpBuf, tiling.baseM * tiling.baseN * sizeof(cType));

    SetSysWorkspace(workspace);
    printf("core=%u, valid=%d, offsetA=%d, offsetB=%d, offsetC=%d, offsetBias=%d, singleCoreM=%d, singleCoreN=%d, Ka=%d, Kb=%d, N=%d\n",
           AscendC::GetBlockIdx(), (int)isValidBlock, offsetA, offsetB, offsetC, offsetBias,
           tiling.singleCoreM, tiling.singleCoreN, tiling.Ka, tiling.Kb, tiling.N);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulPreluKernel<aType, bType, cType, biasType>::Process(AscendC::TPipe *pipe)
{
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }

    uint32_t computeRound = 0;
    REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), matmulObj);
    matmulObj.Init(&tiling);
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);

    while (matmulObj.template Iterate<true>()) {
        MatmulCompute();
        PreluCompute();
        CopyOut(computeRound);
        computeRound++;
    }
    matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulPreluKernel<aType, bType, cType, biasType>::MatmulCompute()
{
    reluOutLocal = reluOutQueue.template AllocTensor<cType>();
    matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulPreluKernel<aType, bType, cType, biasType>::PreluCompute()
{
    tmpLocal = tmpBuf.template Get<cType>();

    AscendC::Mins(tmpLocal, reluOutLocal, static_cast<cType>(0), tiling.baseM * tiling.baseN);
    AscendC::Muls(tmpLocal, tmpLocal, alpha, tiling.baseM * tiling.baseN);
    AscendC::Maxs(reluOutLocal, reluOutLocal, static_cast<cType>(0), tiling.baseM * tiling.baseN);
    AscendC::Add(reluOutLocal, reluOutLocal, tmpLocal, tiling.baseM * tiling.baseN);

    reluOutQueue.EnQue(reluOutLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulPreluKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count)
{
    reluOutLocal = reluOutQueue.template DeQue<cType>();

    // 非法 block 不写回，避免覆盖正确结果
    if (!isValidBlock) {
        reluOutQueue.FreeTensor(reluOutLocal);
        return;
    }

    const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
    uint32_t startOffset = (count / roundN) * tiling.baseM * tiling.N
                         + (count % roundN) * tiling.baseN;

    AscendC::DataCopyParams copyParam = {(uint16_t)tiling.baseM,
                                         (uint16_t)(tiling.baseN * sizeof(cType) / DEFAULT_C0_SIZE),
                                         0,
                                         (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / DEFAULT_C0_SIZE)};
    AscendC::DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
    reluOutQueue.FreeTensor(reluOutLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulPreluKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                             int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                                             int32_t &offsetBias)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto nSingleBlocks = Ceiling(tiling.N, tiling.singleCoreN);
    auto validBlocks = mSingleBlocks * nSingleBlocks;

    // MIX 模式下 AIV block 可能多于实际 tiling block，超出的 block 做安全映射
    isValidBlock = (blockIdx < validBlocks);

    if (!isValidBlock) {
        // 非法 block 映射到 0 号块，避免越界
        blockIdx = 0;
    }

    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void matmul_prelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                          float alpha, GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    MatmulPreluKernel<half, half, float, float> kernel;
    kernel.Init(a, b, bias, c, alpha, workspace, tiling, &pipe);
    kernel.Process(&pipe);
}