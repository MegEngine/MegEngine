/*
 *   LIBCUDA_PATH: candidate paths to libcuda.so; multiple paths are
 *   splitted by colons
 **/

#pragma GCC visibility push(default)

#include <cstdio>
#define LOGE(fmt, v...) fprintf(stderr, "err: " fmt "\n", ##v)

extern "C" {
#include <cuda.h>
}
#include <cudaProfiler.h>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

static const char* default_so_paths[] = {
    "/usr/local/nvidia/lib64/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "libcuda.so",
};

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#define F_OK 0
#define RTLD_LAZY 0
// On the windows platform we use a lib_filename without a full path so
// the win-api "LoadLibrary" would uses a standard search strategy to
// find the lib module. As we cannot access to the lib_filename without a
// full path, we should not use "access(a, b)" to verify it.
#define access(a, b) false

static void* dlopen(const char* file, int) {
    return static_cast<void*>(LoadLibrary(file));
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

static bool open_shared_lib(const char* path, void*& handle) {
    if (!access(path, F_OK)) {
        handle = dlopen(path, RTLD_LAZY);
        if (handle)
            return true;
        LOGE("cuda lib found but can not be opened: %s err=%s", path,
             dlerror());
    }
    return false;
}

static void* get_library_handle() {
    const char* path = nullptr;
    auto str_cptr = getenv("LIBCUDA_PATH");
    std::string str;
    void* handle = nullptr;

    if (str_cptr) {
        str = str_cptr;
        char* p = &str[0];
        const char* begin = p;
        while (*p) {
            if (*p == ':') {
                *p = 0;
                if (open_shared_lib(begin, handle)) {
                    path = begin;
                    break;
                }
                begin = p + 1;
            }
            ++p;
        }
        if (open_shared_lib(begin, handle)) {
            path = begin;
        }
    }

    if (!path) {
        for (size_t i = 0; i < (sizeof(default_so_paths) / sizeof(char*));
             i++) {
            if (open_shared_lib(default_so_paths[i], handle)) {
                path = default_so_paths[i];
                break;
            }
        }
    }

    if (!path) {
        LOGE("can not find cuda");
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

