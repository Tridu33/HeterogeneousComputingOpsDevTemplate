#include "tiling_api.h"
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <cassert>
using namespace matmul_tiling;
using namespace std;

void WriteTilingFile(optiling::TCubeTiling *tilingData) {
    uint32_t tilingSize = tilingData->GetDataSize();
    char *buf = (char *)malloc(tilingSize);
    tilingData->SaveToBuffer(buf, tilingSize);
    ofstream outfile("input/tiling.bin", ios::out | ios::binary);
    if (!outfile) {
        cout << "Failed to open file." << endl;
        return;
    }

    outfile.write(buf, tilingSize);
    outfile.close();
}

int main(int argc, char *argv[])
{
    int M = 512;
    int N = 1024;
    int K = 512;
    TPosition leftPos = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;
    int transposeA = 0;

    TPosition rightPos = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT16;
    int transposeB = 0;

    TPosition resPos = TPosition::GM;
    CubeFormat resFormat = CubeFormat::ND;
    DataType resDtype = DataType::DT_FLOAT;

    TPosition biasPos = TPosition::GM;
    CubeFormat biasFormat = CubeFormat::ND;
    DataType biasDtype = DataType::DT_FLOAT;
    int isBias = 0;

    int usedCoreNum = 1;
    int runMode = 0;

    // single core mode: runMode = 0
    // multi core mode: runMode = 1
    if (runMode == 0) {
        optiling::TCubeTiling tilingData;
        tilingData.set_usedCoreNum(usedCoreNum);
        MatmulApiTiling tilingApi;
        tilingApi.SetAType(leftPos, leftFormat, leftDtype, bool(transposeA));
        tilingApi.SetBType(rightPos, rightFormat, rightDtype, bool(transposeB));
        tilingApi.SetCType(resPos, resFormat, resDtype);
        tilingApi.SetBiasType(biasPos, biasFormat, biasDtype);

        tilingApi.SetShape(M, N, K);
        tilingApi.SetOrgShape(M, N, K);
        tilingApi.SetBias(bool(isBias));

        tilingApi.SetBufferSpace(-1, -1, -1);
        int64_t res = tilingApi.GetTiling(tilingData);
        if (res == -1) {
            std::cout << "gen tiling failed" << std::endl;
        } else {
            WriteTilingFile(&tilingData);
        }
    } else if (runMode = 1) {
        optiling::TCubeTiling tilingData;
        tilingData.set_usedCoreNum(usedCoreNum);
        MultiCoreMatmulTiling tilingApi;
        tilingApi.SetDim(usedCoreNum);
        tilingApi.SetAType(leftPos, leftFormat, leftDtype, bool(transposeA));
        tilingApi.SetBType(rightPos, rightFormat, rightDtype, bool(transposeB));
        tilingApi.SetCType(resPos, resFormat, resDtype);
        tilingApi.SetBiasType(biasPos, biasFormat, biasDtype);

        tilingApi.SetOrgShape(M, N, K);
        tilingApi.SetShape(M, N, K);
        tilingApi.SetBias(bool(isBias));

        tilingApi.SetBufferSpace(-1, -1, -1);
        int64_t res = tilingApi.GetTiling(tilingData);
        if (res == -1) {
            std::cout << "gen tiling failed" << std::endl;
        } else {
            WriteTilingFile(&tilingData);
        }
    }
}
