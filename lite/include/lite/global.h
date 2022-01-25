/**
 * \file inlude/lite/global.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "macro.h"
#include "network.h"

#include <functional>
#include <memory>
#include <vector>

namespace lite {

/**
 * \brief Model decryption function
 *
 * \param[in] const void* is the decrypted model memory pointer
 * \param[in] size_t the size the decrypted model memory in byte
 * \param[in] const std::vector<uint8_t>& the decryption key vector
 */
using DecryptionFunc = std::function<std::vector<uint8_t>(
        const void*, size_t, const std::vector<uint8_t>&)>;

/**
 * \brief register a custom decryption method and key to lite.
 *
 * \param[in] decrypt_name the name of the decryption, which will act as the
 * hash key to find the decryption method.
 *
 * \param[in] func the decryption function, which will decrypt the model with
 * the registered key, return a vector that contain the decrypted model.
 *
 * \param[in] key the decryption key of the method
 */
LITE_API bool register_decryption_and_key(
        std::string decrypt_name, const DecryptionFunc& func,
        const std::vector<uint8_t>& key);

/**
 * \brief update decryption function or key of a custom decryption method.
 *
 * \param[in] decrypt_name the name of the decryption, which will act as the
 * hash key to find the decryption method.
 *
 * \param[in] func the decryption function, which will decrypt the model with
 * the registered key, return a vector that contain the decrypted model. if
 * function is nullptr, it will not be updated.
 *
 * \param[in] key the decryption key of the method, if the size of key is zero,
 * it will not be updated
 */
LITE_API bool update_decryption_or_key(
        std::string decrypt_name, const DecryptionFunc& func,
        const std::vector<uint8_t>& key);

/**
 * \brief Model information parse function
 *
 * \param[in] const void* is the information memory
 * \param[in] size_t the size the information memory
 * \param[in] const std::string the model name used for check whether the
 * infomation match the model
 * \param[in] Config the model config, ParseInfoFunc can fill it with the
 * information in json, the config will influence Network loading later
 * \param[in] NetworkIO the model IO, ParseInfoFunc can fill it with the
 * information in json, the networkio will influence Network forwarding later
 * \param[in] std::unordered_map<std::string, LiteAny>& isolated_config_map, the
 * other config not inclue in config and networkIO, ParseInfoFunc can fill it
 * with the information in json, now support:
 * "device_id" : int, default 0
 * "number_threads" : uint32_t, default 1
 * "is_inplace_model" : bool, default false
 * "use_tensorrt" : bool, default false
 */
using ParseInfoFunc = std::function<bool(
        const void*, size_t, const std::string model_name, Config& config,
        NetworkIO& network_io,
        std::unordered_map<std::string, LiteAny>& isolated_config_map,
        std::string& extra_info)>;

/**
 * \brief register a custom parser function to lite.
 *
 * \param[in] info_type the name of the parser function, which will act as the
 * hash key to find the parser method.
 *
 * \param[in] parse_func the parser function, which will parse the given
 * information and modify the Network Config and IO.
 *
 */
LITE_API bool register_parse_info_func(
        std::string info_type, const ParseInfoFunc& parse_func);

/*! \brief Get version
 */
LITE_API void get_version(int& major, int& minor, int& patch);

/*! \brief Set the current log level.
 * \param[in] level The new log level
 */
LITE_API void set_log_level(LiteLogLevel level);

/*! \brief Get the current log level.
 * \return The current log level
 */
LITE_API LiteLogLevel get_log_level();

/*! \brief Get device count
 * \param[in] device_type device type
 * \return the device count
 */
LITE_API size_t get_device_count(LiteDeviceType device_type);

/*! \brief try to coalesce all free memory in megenine
 */
LITE_API void try_coalesce_all_free_memory();

/*!
 * \brief Set the loader to the lite
 * \param loader_path is the file path which store the cache
 */
LITE_API void set_loader_lib_path(const std::string& loader_path);

/*!
 * \brief Set the algo policy cache file for CPU/CUDA ...
 * \param cache_path is the file path which store the cache
 * \param always_sync sync the cache when model run
 */
LITE_API void set_persistent_cache(
        const std::string& cache_path, bool always_sync = false);

/*!
 * \brief dump the PersistentCache policy cache to file, if the network is set
 * to profile when forward, though this the algo policy will dump to file
 */
LITE_API void dump_persistent_cache(const std::string& cache_path);

/*!
 * \brief Set the TensorRT engine cache path for serialized prebuilt ICudaEngine
 */
LITE_API void set_tensor_rt_cache(std::string tensorrt_cache_path);

/*!
 * \brief dump the TensorRT cache to the file set in set_tensor_rt_cache
 */
LITE_API void dump_tensor_rt_cache();

/**
 * register the physical and virtual address pair to the mge, some device
 * need the map from physical to virtual.
 */
LITE_API bool register_memory_pair(
        void* vir_ptr, void* phy_ptr, size_t length, LiteDeviceType device,
        LiteBackend backend = LiteBackend::LITE_DEFAULT);

/**
 * clear the physical and virtual address pair in mge.
 */
LITE_API bool clear_memory_pair(
        void* vir_ptr, void* phy_ptr, LiteDeviceType device,
        LiteBackend backend = LiteBackend::LITE_DEFAULT);

}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
