/**
 * @file matmul_prelu_custom_tiling.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;
using namespace std;

uint8_t *GetTilingBuf(optiling::TCubeTiling *tilingData)
{
    uint32_t tilingSize = tilingData->GetDataSize();
    uint8_t *buf = (uint8_t *)malloc(tilingSize);
    tilingData->SaveToBuffer(buf, tilingSize);
    return buf;
}

uint8_t *GenerateTiling(const char *socVersion, uint32_t &blockDim)
{
    int M = 1024;
    int N = 640;
    int K = 256;

    TPosition leftPosition = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;
    bool isTransA = false;

    TPosition rightPosition = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT16;
    bool isTransB = false;

    // 改这里：先用 GM 验证
    TPosition resultPosition = TPosition::GM;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_FLOAT;

    TPosition biasPosition = TPosition::GM;
    CubeFormat biasFormat = CubeFormat::ND;
    DataType biasDtype = DataType::DT_FLOAT;
    bool isBias = true;

    int usedCoreNum = 2;
    int baseM = 128;
    int baseN = 128;

    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);
    tilingApi.SetBiasType(biasPosition, biasFormat, biasDtype);

    tilingApi.SetShape(M, N, K);

    // 高度怀疑这里第四个参数应该不是 K，先按常见情况尝试 N
    tilingApi.SetOrgShape(M, N, K, N);

    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetSingleShape(M / usedCoreNum, N, K);
    tilingApi.SetFixSplit(baseM, baseN, -1);

    int64_t res = tilingApi.GetTiling(tilingData);
    
    tilingData.set_stepM(4);   // 应该是 4
    tilingData.set_stepN(5);   // 应该是 5

    int32_t dim = 0;
    int32_t mDim = 0;
    int32_t nDim = 0;
    tilingApi.GetCoreNum(dim, mDim, nDim);
    blockDim = static_cast<uint32_t>(dim);

    std::cout << "blockDim=" << blockDim
          << ", mDim=" << mDim
          << ", nDim=" << nDim
          << ", M=" << M
          << ", N=" << N
          << ", K=" << K
          << ", singleCoreM=" << tilingData.get_singleCoreM()
          << ", singleCoreN=" << tilingData.get_singleCoreN()
          << ", singleCoreK=" << tilingData.get_singleCoreK()
          << ", baseM=" << tilingData.get_baseM()
          << ", baseN=" << tilingData.get_baseN()
          << ", stepM=" << tilingData.get_stepM()
          << ", stepN=" << tilingData.get_stepN()
          << std::endl;

    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    return GetTilingBuf(&tilingData);
}