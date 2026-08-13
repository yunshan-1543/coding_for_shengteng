#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <securec.h>

#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 1
#endif

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 0)
    #undef ASCENDC_DUMP
#endif

static char ascendcErrMsg[1024] = {0};

static void *g_kernel_handle = nullptr;

struct ascend_kernels {
    uint32_t version;
    uint32_t type_cnt;
    uint32_t mix_type;
    uint32_t mix_len;
    uint32_t mix_file_len;
    uint8_t mix_buf[1025048];
} __ascend_kernel_ascend910b1_ascendc_kernels_npu __attribute__ ((section (".ascend.kernel.ascend910b1.ascendc_kernels_npu"))) = {1,1,0,1025048,1025048,{0}};

extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim, void **args,
                            uint32_t size, const void *stream);
uint32_t GetAscendCoreSyncAddr(void **addr);
int UnregisterAscendBinary(void *hdl);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
bool AscendCheckSoCVersion(const char *socVersion, char* errMsg);
void AscendProfRegister();
uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);
uint32_t LaunchAscendKernelForVectorCore(const char *opType, void *handle, const uint64_t key, void **args, uint32_t size,
    const void *stream, bool enbaleProf, uint32_t aicBlockDim, uint32_t aivBlockDim, uint32_t aivBlockDimOffset);
}

namespace Adx {
    void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
                            void *stream, const char *opType);
}

    class KernelHandleGradUnregister {
    private:
        KernelHandleGradUnregister() {}

    public:
        KernelHandleGradUnregister(const KernelHandleGradUnregister&) = delete;
        KernelHandleGradUnregister& operator=(const KernelHandleGradUnregister&) = delete;

        static KernelHandleGradUnregister& GetInstance() {
            static KernelHandleGradUnregister instance;
            return instance;
        }
        ~KernelHandleGradUnregister(){
            if (g_kernel_handle) {
                UnregisterAscendBinary(g_kernel_handle);
                g_kernel_handle = nullptr;
            }
        }
    };

static void __register_kernels(void) __attribute__((constructor));
void __register_kernels(void)
{
    const char* compileSocVersion = "ascend910b1";
    uint32_t ret;

    bool checkSocVersion = AscendCheckSoCVersion(compileSocVersion, ascendcErrMsg);
    if (!checkSocVersion) {
        return;
    }
    ret = RegisterAscendBinary(
        (const char *)__ascend_kernel_ascend910b1_ascendc_kernels_npu.mix_buf,
        __ascend_kernel_ascend910b1_ascendc_kernels_npu.mix_file_len,
        0,
        &g_kernel_handle);
    if (ret != 0) {
        printf("RegisterAscendBinary mix ret %u \n", ret);
    }

    AscendProfRegister();
}





uint32_t launch_and_profiling_matmul_prelu_custom(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "matmul_prelu_custom";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 0, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_matmul_prelu_custom(uint32_t blockDim, void* stream, void* a, void* b, void* bias, void* c, float alpha, void* workspace, void* tilingGm)
{
    struct {
    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
            void* __ascendc_dump;
    #endif
        alignas(((alignof(void*) + 3) >> 2) << 2) void* ffts_addr;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* a;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* b;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* bias;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* c;
        alignas(((alignof(float) + 3) >> 2) << 2) float alpha;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* workspace;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* tilingGm;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    constexpr uint32_t __ascendc_one_core_dump_size = 1048576;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_dump), __ascendc_one_core_dump_size * 75);
#endif
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);

    void *ffts_addr;
    __ascendc_ret = GetAscendCoreSyncAddr(&ffts_addr);
    if (__ascendc_ret != 0) {
        printf("GetAscendCoreSyncAddr ret %u\n", __ascendc_ret);
        return __ascendc_ret;
    }

    __ascendc_args.ffts_addr = ffts_addr;
    __ascendc_args.a = a;
    __ascendc_args.b = b;
    __ascendc_args.bias = bias;
    __ascendc_args.c = c;
    __ascendc_args.alpha = alpha;
    __ascendc_args.workspace = workspace;
    __ascendc_args.tilingGm = tilingGm;

    const char *__ascendc_name = "matmul_prelu_custom";
    __ascendc_ret = launch_and_profiling_matmul_prelu_custom(0, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    Adx::AdumpPrintWorkSpace(__ascendc_args.__ascendc_dump, __ascendc_one_core_dump_size * 75, stream, __ascendc_name);
    FreeAscendMemDevice(__ascendc_args.__ascendc_dump);
#endif
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}
