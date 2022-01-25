/**
 * \file src/global.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <lite_build_config.h>

#include "decryption/aes_decrypt.h"
#include "decryption/decrypt_base.h"
#include "decryption/rc4_cryption.h"
#include "lite/global.h"
#include "misc.h"
#include "parse_info/default_parse.h"
#include "parse_info/parse_info_base.h"

#if LITE_BUILD_WITH_MGE
#include "megbrain/common.h"
#include "megbrain/comp_node.h"
#include "megbrain/serialization/extern_c_opr.h"
#include "megbrain/version.h"
#include "megbrain/utils/infile_persistent_cache.h"
#include "mge/common.h"
#if MGB_ENABLE_TENSOR_RT
#include "megbrain/tensorrt/tensorrt_engine_cache.h"
#endif
#endif

#include <mutex>
#include <unordered_map>

using namespace lite;

lite::DecryptionStaticData& lite::decryption_static_data() {
    static lite::DecryptionStaticData global_map;
    return global_map;
}

void lite::get_version(int& major, int& minor, int& patch) {
#if LITE_BUILD_WITH_MGE
    auto version = mgb::get_version();
    major = version.major;
    minor = version.minor;
    patch = version.patch;
#else
    //! without mge, the version set the max version
    major = 8;
    minor = 9999;
    patch = 0;
#endif
}

size_t lite::get_device_count(LiteDeviceType device_type) {
#if LITE_BUILD_WITH_MGE
    auto mgb_device_type = to_compnode_locator(device_type).type;
    return mgb::CompNode::get_device_count(mgb_device_type);
#else
    LITE_MARK_USED_VAR(device_type);
    LITE_THROW("no lite backend avialible, please check build macro.");
#endif
}

bool lite::register_decryption_and_key(
        std::string decrypt_name, const DecryptionFunc& func,
        const std::vector<uint8_t>& key) {
    LITE_LOCK_GUARD(decryption_static_data().map_mutex);
    auto& global_map = decryption_static_data().decryption_methods;
    if (global_map.find(decrypt_name) != global_map.end()) {
        LITE_THROW(ssprintf(
                "The decryption method %s is already registered.",
                decrypt_name.c_str()));
        return false;
    } else {
        auto key_pointer = std::make_shared<std::vector<uint8_t>>(key);
        global_map[decrypt_name] = {func, key_pointer};
        LITE_LOG("Registered ecryption method %s.", decrypt_name.c_str());
        return true;
    }
}

bool lite::update_decryption_or_key(
        std::string decrypt_name, const DecryptionFunc& func,
        const std::vector<uint8_t>& key) {
    LITE_LOCK_GUARD(decryption_static_data().map_mutex);
    auto& global_map = decryption_static_data().decryption_methods;
    if (global_map.find(decrypt_name) != global_map.end()) {
        std::shared_ptr<std::vector<uint8_t>> key_pointer;
        DecryptionFunc new_func;
        if (func) {
            new_func = func;
            LITE_LOG("%s decryption function is updated.", decrypt_name.c_str());
        } else {
            new_func = global_map[decrypt_name].first;
        }
        if (key.size()) {
            key_pointer = std::make_shared<std::vector<uint8_t>>(key);
            LITE_LOG("%s decryption key is updated.", decrypt_name.c_str());
        } else {
            key_pointer = global_map[decrypt_name].second;
        }
        global_map[decrypt_name] = {new_func, key_pointer};
        return true;
    } else {
        LITE_THROW(ssprintf(
                "The decryption method %s is not registered.", decrypt_name.c_str()));
        return false;
    }
}

lite::ParseInfoStaticData& lite::parse_info_static_data() {
    static lite::ParseInfoStaticData global_map;
    return global_map;
}

bool lite::register_parse_info_func(
        std::string info_type, const ParseInfoFunc& parse_func) {
    LITE_LOCK_GUARD(parse_info_static_data().map_mutex);
    auto& global_map = parse_info_static_data().parse_info_methods;
    if (global_map.find(info_type) != global_map.end()) {
        LITE_THROW(ssprintf(
                "The parse info method %s is already registered.", info_type.c_str()));
        return false;
    } else {
        global_map[info_type] = parse_func;
        LITE_LOG("Registered infomation parser method %s.", info_type.c_str());
        return true;
    }
}

#if LITE_BUILD_WITH_MGE

namespace {
struct CacheControl {
    LITE_MUTEX cache_mutex;
    std::string cache_type = "file";
    std::atomic_size_t config_algo_times{0};
    std::atomic_size_t config_trt_times{0};
};
CacheControl cache_control;
}  // namespace

void lite::try_coalesce_all_free_memory() {
    mgb::CompNode::try_coalesce_all_free_memory();
}

void lite::set_loader_lib_path(const std::string& loader_path) {
    const char* lib_path = loader_path.c_str();
    LITE_LOG("load a device loader of path %s.", lib_path);
    auto handle = dlopen(lib_path, RTLD_LAZY);
    LITE_ASSERT(handle, "failed to open c opr lib %s: %s", lib_path, dlerror());
    const char* entry = MGB_C_OPR_INIT_FUNC_STR;
    auto func = dlsym(handle, entry);
    LITE_ASSERT(func, "can not resolve %s: %s", entry, dlerror());
    typedef void (*entry_f_t)(void*);
    reinterpret_cast<entry_f_t>(func)(
            reinterpret_cast<void*>(&mgb_get_extern_c_opr_api_versioned));
}

void lite::set_persistent_cache(const std::string& cache_path, bool always_sync) {
    LITE_LOCK_GUARD(cache_control.cache_mutex);
    cache_control.cache_type = "file";
    if (cache_control.config_algo_times >= 1) {
        LITE_WARN(
                "The cache has been set，maybe some model is using now, change "
                "it now may cause unknow error!!");
    }
    cache_control.config_algo_times++;
    mgb::PersistentCache::set_impl(std::make_shared<mgb::InFilePersistentCache>(
            cache_path.c_str(), always_sync));
}

void lite::dump_persistent_cache(const std::string& cache_path) {
    LITE_LOCK_GUARD(cache_control.cache_mutex);
    LITE_ASSERT(
            cache_control.cache_type == "file",
            "now cache type not correct, it can't be dumped.");
    static_cast<mgb::InFilePersistentCache&>(mgb::PersistentCache::inst())
            .dump_cache(cache_path.c_str());
}

//! Set the TensorRT engine cache path for serialized prebuilt ICudaEngine
void lite::set_tensor_rt_cache(std::string tensorrt_cache_path) {
#if MGB_ENABLE_TENSOR_RT
    LITE_LOCK_GUARD(cache_control.cache_mutex);
    if (cache_control.config_trt_times >= 1) {
        LITE_WARN(
                "The trt cache has been set，maybe some model is using now, "
                "change it now may cause unknow error!!");
    }
    cache_control.config_trt_times++;
    mgb::TensorRTEngineCache::enable_engine_cache(true);
    mgb::TensorRTEngineCache::set_impl(
            std::make_shared<mgb::TensorRTEngineCacheIO>(tensorrt_cache_path));
#else
    LITE_MARK_USED_VAR(tensorrt_cache_path);
    LITE_THROW("TensorRT is disable at compile time.");
#endif
}

void lite::dump_tensor_rt_cache() {
#if MGB_ENABLE_TENSOR_RT
    if (mgb::TensorRTEngineCache::enable_engine_cache()) {
        mgb::TensorRTEngineCache::inst().dump_cache();
    }
#else
    LITE_THROW("TensorRT is disable at compile time.");
#endif
}

bool lite::register_memory_pair(
        void* vir_ptr, void* phy_ptr, size_t length, LiteDeviceType device,
        LiteBackend backend) {
    LITE_MARK_USED_VAR(vir_ptr);
    LITE_MARK_USED_VAR(phy_ptr);
    LITE_MARK_USED_VAR(length);
    LITE_MARK_USED_VAR(device);
    LITE_MARK_USED_VAR(backend);
    LITE_THROW("register_memory_pair is not implement yet!");
}

bool lite::clear_memory_pair(
        void* vir_ptr, void* phy_ptr, LiteDeviceType device, LiteBackend backend) {
    LITE_MARK_USED_VAR(vir_ptr);
    LITE_MARK_USED_VAR(phy_ptr);
    LITE_MARK_USED_VAR(device);
    LITE_MARK_USED_VAR(backend);
    LITE_THROW("clear_memory_pair is not implement yet!");
}

#else  // LITE_BUILD_WITH_MGE
void lite::try_coalesce_all_free_memory() {}

void lite::set_loader_lib_path(const std::string&) {
    LITE_THROW("mge is disbale at build time, please build with mge");
}

void lite::set_persistent_cache(const std::string&, bool) {
    LITE_THROW("mge is disbale at build time, please build with mge");
}

void lite::dump_persistent_cache(const std::string&) {
    LITE_THROW("mge is disbale at build time, please build with mge");
}

//! Set the TensorRT engine cache path for serialized prebuilt ICudaEngine
void lite::set_tensor_rt_cache(std::string) {
    LITE_THROW("mge is disbale at build time, please build with mge");
}

void lite::dump_tensor_rt_cache() {
    LITE_THROW("mge is disbale at build time, please build with mge");
}

bool lite::register_memory_pair(
        void* vir_ptr, void* phy_ptr, size_t length, LiteDeviceType device,
        LiteBackend beckend) {
    LITE_THROW("register_memory_pair is not implement yet!");
}

bool lite::clear_memory_pair(
        void* vir_ptr, void* phy_ptr, LiteDeviceType device, LiteBackend beckend) {
    LITE_THROW("clear_memory_pair is not implement yet!");
}
#endif
namespace lite {
REGIST_DECRYPTION_METHOD(
        "AES_default", lite::AESDcryption::decrypt_model,
        lite::AESDcryption::get_decrypt_key());

REGIST_DECRYPTION_METHOD(
        "RC4_default", lite::RC4::decrypt_model, lite::RC4::get_decrypt_key());

REGIST_DECRYPTION_METHOD(
        "SIMPLE_FAST_RC4_default", lite::SimpleFastRC4::decrypt_model,
        lite::SimpleFastRC4::get_decrypt_key());

REGIST_PARSE_INFO_FUNCTION("LITE_default", lite::default_parse_info);
}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
