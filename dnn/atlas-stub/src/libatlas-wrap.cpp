/**
 * \file dnn/atlas-stub/src/libatlas-wrap.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma GCC visibility push(default)

#include <cstdio>
#define LOGE(fmt, v...) fprintf(stderr, "err: " fmt "\n", ##v)

#include "acl/acl.h"

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
float on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return 0.f;
}
template <>
aclFloat16 on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return 0;
}
template <>
aclDataBuffer* on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return nullptr;
}
template <>
aclError on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return ACL_ERROR_INTERNAL_ERROR;
}
template <>
void* on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return nullptr;
}
template <>
uint32_t on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return 0;
}
template <>
size_t on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return 0;
}
template <>
void on_init_failed(int func_idx) {
    log_failed_load(func_idx);
}
template <>
int64_t on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return 0;
}
template <>
const char* on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return "load lib failed";
}
template <>
aclopAttr* on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return nullptr;
}
template <>
aclmdlDesc* on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return nullptr;
}
template <>
aclmdlDataset* on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return nullptr;
}
template <>
aclFormat on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return ACL_FORMAT_UNDEFINED;
}
template <>
aclTensorDesc* on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return nullptr;
}
template <>
aclDataType on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return ACL_DT_UNDEFINED;
}
template <>
aclmdlAIPP* on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return nullptr;
}
}  // namespace

#include "./libatlas-wrap.h"

static const char* default_so_paths[] = {
        "/usr/local/Ascend/acllib/lib64/libascendcl.so",
        "libascendcl.so",
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
        LOGE("Failed to load atlas library");
        return nullptr;
    }
    return handle;
}

static void log_failed_load(int func_idx) {
    LOGE("failed to load atlas func: %s", g_func_name[func_idx]);
}

static void* resolve_library_func(void* handle, const char* func) {
    if (!handle) {
        LOGE("handle should not be nullptr!");
        return nullptr;
    }
    auto ret = dlsym(handle, func);
    if (!ret) {
        LOGE("failed to load atlas func: %s", func);
    }
    return ret;
}
