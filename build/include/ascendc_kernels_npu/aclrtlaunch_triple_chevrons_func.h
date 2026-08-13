
#ifndef HEADER_ACLRTLAUNCH_MATMUL_PRELU_CUSTOM_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_MATMUL_PRELU_CUSTOM_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_matmul_prelu_custom(uint32_t blockDim, void* stream, void* a, void* b, void* bias, void* c, float alpha, void* workspace, void* tilingGm);

inline uint32_t matmul_prelu_custom(uint32_t blockDim, void* hold, void* stream, void* a, void* b, void* bias, void* c, float alpha, void* workspace, void* tilingGm)
{
    (void)hold;
    return aclrtlaunch_matmul_prelu_custom(blockDim, stream, a, b, bias, c, alpha, workspace, tilingGm);
}

#endif
