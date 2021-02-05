#pragma GCC visibility push(default)

#include <cstdio>
#define LOGE(fmt, v...) fprintf(stderr, "err: " fmt "\n", ##v)


extern "C" {
#include "cuda.h"
}
#include "cudaProfiler.h"

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

static void log_failed_load(int func_idx);
namespace {
template <typename T>
T on_init_failed(int func_idx);
template <>
CUresult on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return CUDA_ERROR_UNKNOWN;
}

}  // namespace

#define _WRAPLIB_API_CALL CUDAAPI
#define _WRAPLIB_CALLBACK CUDA_CB

#if CUDA_VERSION == 10010
#include "./libcuda-wrap_10.1.h"
#elif CUDA_VERSION == 10020
#include "./libcuda-wrap_10.2.h"
#elif CUDA_VERSION == 11010
#include "./libcuda-wrap_11.1.h"
#elif CUDA_VERSION == 11020
#include "./libcuda-wrap_11.2.h"
#else
#error "cuda stub not support this cuda version, you can close cuda stub to passby"
#endif


#undef _WRAPLIB_CALLBACK
#undef _WRAPLIB_API_CALL

static const char* default_so_name =
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        "nvcuda.dll";
#elif defined(__APPLE__) || defined(__MACOSX)
        "libcuda.dylib";
#else
        "libcuda.so.1";
#endif

// Harvested from cuda_drvapi_dynlink.c
static const char* default_so_paths[] = {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        "nvcuda.dll",
#elif defined(__unix__) || defined(__QNX__) || defined(__APPLE__) || \
        defined(__MACOSX)
#if defined(__APPLE__) || defined(__MACOSX)
        "/usr/local/cuda/lib/libcuda.dylib",
#elif defined(__ANDROID__)
#if defined(__aarch64__)
        "/system/vendor/lib64/libcuda.so",
#elif defined(__arm__)
        "/system/vendor/lib/libcuda.so",
#endif
#else
        "libcuda.so.1",
#endif
#else
#error "Unknown platform"
#endif
};

static const char* extra_so_paths[] = {
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/local/nvidia/lib64/libcuda.so",
};

static const char* g_default_api_name = "cuda";
#include "./dlopen_helper.h"