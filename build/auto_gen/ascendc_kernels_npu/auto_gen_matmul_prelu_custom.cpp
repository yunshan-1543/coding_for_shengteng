#ifndef __MATMUL_PRELU_CUSTOM__KERNEL_FUN_H__
#define __MATMUL_PRELU_CUSTOM__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define matmul_prelu_custom matmul_prelu_custom_origin
#include "/home/fanyi/fyfy/matmul_prelu_custom.cpp"

#undef matmul_prelu_custom
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_matmul_prelu_custom_kernel(
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
GM_ADDR dumpAddr,
#endif
GM_ADDR ffts_addr, __attribute__((cce_global)) uint8_t* a, __attribute__((cce_global)) uint8_t* b, __attribute__((cce_global)) uint8_t* bias, __attribute__((cce_global)) uint8_t* c, float alpha, __attribute__((cce_global)) uint8_t* workspace, __attribute__((cce_global)) uint8_t* tilingGm, GM_ADDR overflow_status) {
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    AscendC::InitDump(true, dumpAddr, ONE_CORE_DUMP_SIZE);
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_INIT_DUMP));
#endif
#endif

    icache_preload(1);
    if (ffts_addr != nullptr) {
        set_ffts_base_addr((uint64_t)ffts_addr);
    }
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_FFTS_ADDR));
#endif
#ifdef ASCENDC_DUMP
    uint64_t __ascendc_tStamp = 0;
    uint64_t __ascendc_version = 0;
     __gm__ char* __ascendc_versionStr = nullptr;
    GetCannVersion(__ascendc_versionStr, __ascendc_version, __ascendc_tStamp);
    if (__ascendc_tStamp == 0) {
        AscendC::printf("[WARNING]: CANN TimeStamp is invalid, CANN TimeStamp is %u\n", __ascendc_tStamp);
    } else {
        AscendC::printf("CANN Version: %s, TimeStamp: %u\n", (__gm__ const char*)(__ascendc_versionStr), __ascendc_tStamp);
    }
#endif
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = workspace;
#else
    workspace_param = tilingGm;
#endif
    if (workspace_param == nullptr) {
        return;
    }
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(REGIST_MATMUL_OBJ) || defined(__MIX_CORE_MACRO__)
    if constexpr (g_coreType == AscendC::AIC) {
        matmul::clearWorkspace(workspace_param);
#ifdef ASCENDC_TIME_STAMP_ON
        AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_CLEAR_WK_SPAC));
#endif
    }
#endif
#if defined(HAVE_TILING)
    workspace = workspace_usr;
#else
    tilingGm = workspace_usr;
#endif
#endif
    matmul_prelu_custom_origin(a, b, bias, c, alpha, workspace, tilingGm);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
}

#endif
