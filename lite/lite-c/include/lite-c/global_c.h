/**
 * \file lite-c/include/lite-c/global-c.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#ifndef LITE_C_GLOBAL_H_
#define LITE_C_GLOBAL_H_

#include "macro.h"
#include "network_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Get version
 */
LITE_API int LITE_get_version(int* major, int* minor, int* patch);

/*! \brief Get the last error message.
 * \return the message pointer
 */
LITE_API const char* LITE_get_last_error();

/*! \brief Get device count
 * \param[in] device_type device type
 * \return the device count
 */
LITE_API int LITE_get_device_count(LiteDeviceType device_type, size_t* count);

/*! \brief try to coalesce all free memory in megenine
 */
LITE_API int LITE_try_coalesce_all_free_memory();

/**
 * \brief Model decryption function
 *
 * \param[in] input_data is the decrypted model memory pointer
 * \param[in] input_size the size the decrypted model memory in byte
 * \param[in] key_data decryption key data
 * \param[in] key_size the size of decryption key data
 * \param[out] output_data the data of decrypted data, if output_data is
 * nullptr, just query the output memory length, else write the decryted data to
 * the output_data
 * \return size of decrypted data
 */
typedef size_t (*LiteDecryptionFunc)(const void* input_data, size_t input_size,
                                     const uint8_t* key_data, size_t key_size,
                                     const void* output_data);

/**
 * \brief Model information parse function
 *
 * \param[in] info_data is the information memory
 * \param[in] info_size the size the information memory
 * \param[in] model_name the model name used for check whether the
 * infomation match the model
 * \param[in] config the model config, ParseInfoFunc can fill it with the
 * information in json, the config will influence Network loading later
 * \param[in] network_io the model IO, ParseInfoFunc can fill it with the
 * information in json, the networkio will influence Network forwarding later
 * \param[in] device_id the address to store device_id, default 0
 * \param[in] nr_threads the address to store nr_threads, default 1
 * \param[in] is_inplace_model the address to store is_cpu_inplace_mode, default
 * \param[in] use_tensorrt the address to store is_cpu_inplace_mode, default
 * false
 */
typedef int (*LiteParseInfoFunc)(const void* info_data, size_t info_size,
                                 const char* model_name, LiteConfig* config,
                                 LiteNetworkIO* network_io, int* device_id,
                                 size_t* nr_threads, int* is_cpu_inplace_mode,
                                 int* use_tensorrt);

/**
 * \brief register a custom decryption method and key to lite.
 *
 * \param[in] decrypt_name the name of the decryption, which will act as the
 * hash key to find the decryption method.
 *
 * \param[in] func the decryption function, which will decrypt the model with
 * the registered key, return a vector that contain the decrypted model.
 * \param[in] key_data the decryption key of the method
 * \param[in] key_size the size of decryption key
 */
LITE_API int LITE_register_decryption_and_key(const char* decrypt_name,
                                              const LiteDecryptionFunc func,
                                              const uint8_t* key_data,
                                              size_t key_size);

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
LITE_API int LITE_update_decryption_or_key(const char* decrypt_name,
                                           const LiteDecryptionFunc func,
                                           const uint8_t* key_data,
                                           size_t key_size);

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
LITE_API int LITE_register_parse_info_func(const char* info_type,
                                           const LiteParseInfoFunc parse_func);

/*!
 * \brief Set the loader to the lite
 * \param[in] loader_path is the file path which store the cache
 */
LITE_API int LITE_set_loader_lib_path(const char* loader_path);

/*!
 * \brief Set the algo policy cache file for CPU/CUDA ...
 * \param[in] cache_path is the file path which store the cache
 * \param[in] always_sync sync the cache when cache updated
 */
LITE_API int LITE_set_persistent_cache(const char* cache_path, int always_sync);

/*!
 * \brief Set the tensor policy cache file for CPU/CUDA ...
 * \param[in] cache_path is the file path which store the cache
 */
LITE_API int LITE_set_tensor_rt_cache(const char* cache_path);

/*! \brief Set the current log level.
 * \param[in] level The new log level
 */
LITE_API int LITE_set_log_level(LiteLogLevel level);

/*! \brief Get the current log level.
 * \param[in] level The pointer to log level
 */
LITE_API int LITE_get_log_level(LiteLogLevel* level);
/*!
 * \brief dump the algo policy cache to file, if the network is set to profile
 * when forward, though this the algo policy will dump to file
 * \param[in] cache_path is the file path which store the cache
 */
LITE_API int LITE_dump_persistent_cache(const char* cache_path);

/*!
 * \brief dump the tensorrt policy cache to file
 */
LITE_API int LITE_dump_tensor_rt_cache();
#endif
#ifdef __cplusplus
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
