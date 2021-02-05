/**
 * \file dnn/cuda-stub/src/libnvrtc.cpp
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
#include "./nvrtc_type.h"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

static void log_failed_load(int func_idx);
namespace {
template <typename T>
T on_init_failed(int func_idx);
template <>
nvrtcResult on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return NVRTC_ERROR_INTERNAL_ERROR;
}
template <>
const char* on_init_failed(int func_idx) {
    log_failed_load(func_idx);
    return "load lib failed";
}
}  // namespace

#include "./libnvrtc-wrap.h"
static const char* default_so_name =
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        "nvrtc.dll";
#elif defined(__APPLE__) || defined(__MACOSX)
        "libnvrtc.dylib";
#else
        "libnvrtc.so";
#endif

static const char* default_so_paths[] = {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        "nvrtc.dll",
#elif defined(__unix__) || defined(__QNX__) || defined(__APPLE__) || \
        defined(__MACOSX)
#if defined(__APPLE__) || defined(__MACOSX)
        "/usr/local/cuda/lib/libnvrtc.dylib",
#elif defined(__ANDROID__)
#if defined(__aarch64__)
        "/system/vendor/lib64/libnvrtc.so",
#elif defined(__arm__)
        "/system/vendor/lib/libnvrtc.so",
#endif
#else
        "libnvrtc.so",

        // In case some users does not have correct search path configured in
        // /etc/ld.so.conf
        "/usr/lib/x86_64-linux-gnu/libnvrtc.so",
        "/usr/local/nvidia/lib64/libnvrtc.so",
        "/usr/local/cuda/lib64/libnvrtc.so",
#endif
#else
#error "Unknown platform"
#endif
};
static const char* extra_so_paths[] = {};

static const char* g_default_api_name = "nvrtc";
#include "./dlopen_helper.h"