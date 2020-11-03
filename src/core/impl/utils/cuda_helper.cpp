/**
 * \file src/core/impl/utils/cuda_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/common.h"
#include "megbrain/exception.h"
#include "megbrain/utils/cuda_helper.h"

#include <set>
#include <fstream>
#include <string>
#include <sstream>

using namespace mgb;

#ifdef WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#include <dlfcn.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifdef WIN32
#define F_OK 0
#define RTLD_LAZY 0
#define RTLD_GLOBAL 0
#define RTLD_NOLOAD 0
#define RTLD_DI_ORIGIN 0
#define access(a, b) false
#define SPLITER ';'
#define PATH_SPLITER '\\'
#define ENV_PATH "Path"
#define NVCC_EXE "nvcc.exe"
void* dlopen(const char* file, int) {
    return static_cast<void*>(LoadLibrary(file));
}

int dlinfo(void* handle, int request, char* path) {
    if (GetModuleFileName((HMODULE)handle, path, PATH_MAX))
        return 0;
    else
        return -1;
}

void* dlerror() {
    const char* errmsg = "dlerror not aviable in windows";
    return const_cast<char*>(errmsg);
}

void* dlsym(void* handle, char* name) {
    FARPROC symbol = GetProcAddress((HMODULE)handle, name);
    return reinterpret_cast<void*>(symbol);
}

int check_file_exist(const char* path, int mode) {
    return _access(path, mode);
}
#else
#define SPLITER ':'
#define PATH_SPLITER '/'
#define ENV_PATH "PATH"
#define NVCC_EXE "nvcc"
int check_file_exist(const char* path, int mode) {
    return access(path, mode);
}
#endif

std::vector<std::string> split_env(const char* env) {
    std::string e(env);
    std::istringstream stream(e);
    std::vector<std::string> ret;
    std::string path;
    while (std::getline(stream, path, SPLITER)) {
        ret.emplace_back(path);
    }
    return ret;
}

//! this function will find file_name in each path in envs. It accepts add
//! intermediate path between env and file_name
std::string find_file_in_envs_with_intmd(
        const std::vector<std::string>& envs, const std::string& file_name,
        const std::vector<std::string>& itmedias = {}) {
    for (auto&& env : envs) {
        auto ret = getenv(env.c_str());
        if (ret) {
            for (auto&& path : split_env(ret)) {
                auto file_path = std::string(path) + PATH_SPLITER + file_name;
                if (!check_file_exist(file_path.c_str(), F_OK)) {
                    return file_path;
                }
                if (!itmedias.empty()) {
                    for (auto&& inter_path : itmedias) {
                        file_path = std::string(path) + PATH_SPLITER + inter_path + PATH_SPLITER +
                                    file_name;
                        if (!check_file_exist(file_path.c_str(), F_OK)) {
                            return file_path;
                        }
                    }
                }
            }
        }
    }
    return std::string{};
}

std::string get_nvcc_root_path() {
    auto nvcc_root_path = find_file_in_envs_with_intmd({ENV_PATH}, NVCC_EXE);
    if (nvcc_root_path.empty()) {
        mgb_throw(MegBrainError,
                  "nvcc not found. Add your nvcc to your environment Path");
    } else {
        auto idx = nvcc_root_path.rfind(PATH_SPLITER);
        return nvcc_root_path.substr(0, idx + 1);
    }
}

std::vector<std::string> mgb::get_cuda_include_path() {
#if MGB_CUDA
    std::vector<std::string> paths;
    // 1. use CUDA_BIN_PATH
    auto cuda_path = getenv("CUDA_BIN_PATH");
    if (cuda_path) {
        paths.emplace_back(std::string(cuda_path) + PATH_SPLITER + "include");
        paths.emplace_back(std::string(cuda_path) + PATH_SPLITER + ".." +
                           PATH_SPLITER + "include");
    }

    // 2. use nvcc path
    auto nvcc_path = get_nvcc_root_path();
    auto cudart_header_path = nvcc_path + ".." + PATH_SPLITER + "include" +
                              PATH_SPLITER + "cuda_runtime.h";
    //! double check path_to_nvcc/../include/cuda_runtime.h exists
    auto ret = check_file_exist(cudart_header_path.c_str(), F_OK);
    if (ret == 0) {
        paths.emplace_back(nvcc_path + "..");
        paths.emplace_back(nvcc_path + ".." + PATH_SPLITER + "include");
    }

    // 3. use libcudart.so library path
    char cuda_lib_path[PATH_MAX];
    auto handle = dlopen("libcudart.so", RTLD_GLOBAL | RTLD_LAZY);
    if(handle != nullptr) {
        mgb_assert(dlinfo(handle, RTLD_DI_ORIGIN, cuda_lib_path) != -1, "%s",
                   dlerror());
        paths.emplace_back(std::string(cuda_lib_path) + PATH_SPLITER + ".." +
                           PATH_SPLITER + "include");
    }
    mgb_assert(paths.size() > 0,
               "can't find cuda include path, check your environment of cuda, "
               "try one of this solutions "
               "1. set CUDA_BIN_PATH to cuda home path "
               "2. add nvcc path in PATH "
               "3. add libcudart.so path in LD_LIBRARY_PATH");
    return paths;
#else
    mgb_throw(MegBrainError, "cuda disabled at compile time");
#endif
}