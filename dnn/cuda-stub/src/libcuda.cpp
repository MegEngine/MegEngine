#pragma GCC visibility push(default)

#include <cstdio>
#define LOGE(fmt, v...) fprintf(stderr, "err: " fmt "\n", ##v)

extern "C" {
#include <cuda.h>
}
#include <cudaProfiler.h>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#if defined(_WIN32)
#include <windows.h>
#define RTLD_LAZY 0

static void* dlopen(const char* file, int) {
    return static_cast<void*>(LoadLibraryA(file));
}

static void* dlerror() {
    const char* errmsg = "dlerror not aviable in windows";
    return const_cast<char*>(errmsg);
}

static void* dlsym(void* handle, const char* name) {
    FARPROC symbol = GetProcAddress((HMODULE)handle, name);
    return reinterpret_cast<void*>(symbol);
}

#else
#include <dlfcn.h>
#include <unistd.h>
#endif

static void log_failed_load(int func_idx);
namespace {
template <typename T>
T on_init_failed(int func_idx);
template <>
CUresult on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return CUDA_ERROR_UNKNOWN;
}
}

#define _WRAPLIB_API_CALL CUDAAPI
#define _WRAPLIB_CALLBACK CUDA_CB
#include "./libcuda-wrap.h"
#undef _WRAPLIB_CALLBACK
#undef _WRAPLIB_API_CALL

// Harvested from cuda_drvapi_dynlink.c
static const char* default_so_paths[] = {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    "nvcuda.dll",
#elif defined(__unix__) || defined (__QNX__) || defined(__APPLE__) || defined(__MACOSX)
#if defined(__APPLE__) || defined(__MACOSX)
    "/usr/local/cuda/lib/libcuda.dylib",
#elif defined(__ANDROID__)
#if defined (__aarch64__)
    "/system/vendor/lib64/libcuda.so",
#elif defined(__arm__)
    "/system/vendor/lib/libcuda.so",
#endif
#else
    "libcuda.so.1",
    
    // In case some users does not have correct search path configured in
    // /etc/ld.so.conf
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/local/nvidia/lib64/libcuda.so",
#endif
#else
#error "Unknown platform"
#endif
};

static void* get_library_handle() {
    void* handle = nullptr;
    for (size_t i = 0; i < (sizeof(default_so_paths) / sizeof(char*)); i++) {
        handle = dlopen(default_so_paths[i], RTLD_LAZY);
        if (handle) {
            break;
        }
    }

    if (!handle) {
        LOGE("Failed to load CUDA Driver API library");
        return nullptr;
    }
    return handle;
}

static void log_failed_load(int func_idx) {
    LOGE("failed to load cuda func: %s", g_func_name[func_idx]);
}

static void* resolve_library_func(void* handle, const char* func) {
    if (!handle) {
        LOGE("handle should not be nullptr!");
        return nullptr;
    }
    auto ret = dlsym(handle, func);
    if (!ret) {
        LOGE("failed to load cuda func: %s", func);
    }
    return ret;
}
