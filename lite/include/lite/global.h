#pragma once

#include "macro.h"
#include "network.h"

#include <functional>
#include <memory>
#include <vector>

namespace lite {

/**
 * @brief Model decryption function
 *
 * @param arg1 the to be decrypted model memory pointer
 * @param arg2 the byte size of the decrypted model memory
 * @param arg3 the decryption key in vector
 * @return the decrypted model in vector format, it's length and content can get by
 * the operators of vector
 */
using DecryptionFunc = std::function<std::vector<uint8_t>(
        const void*, size_t, const std::vector<uint8_t>&)>;

/**
 * @brief register a custom decryption method and key to lite
 *
 * @param decrypt_name the name of the decryption, which will act as the
 * hash key to find the decryption method
 *
 * @param func the decryption function, which will decrypt the model with
 * the registered key, return a vector that contain the decrypted model
 *
 * @param key the decryption key of the method
 *
 * @return Whether or not the decryption method register successful
 */
LITE_API bool register_decryption_and_key(
        std::string decrypt_name, const DecryptionFunc& func,
        const std::vector<uint8_t>& key);

/**
 * @brief update decryption function or key of a custom decryption method, in
 * lite the decryption function and the key store in pair, user can change one of which
 * by this function
 *
 * @param decrypt_name the name of the decryption, which will act as the
 * hash key to find the decryption method
 *
 * @param func the decryption function, which will decrypt the model with
 * the registered key, return a vector that contain the decrypted model. if
 * the function is nullptr, it will not be updated
 *
 * @param key the decryption key of the method, if the size of key is zero,
 * the key will not be updated
 *
 * @return Whether or not the decryption method update successful
 */
LITE_API bool update_decryption_or_key(
        std::string decrypt_name, const DecryptionFunc& func,
        const std::vector<uint8_t>& key);

/**
 * @brief Model information parse function, MegEngine Lite model may pack some
 * information with the model to configure the model inference processing conveniently,
 * this function is used to parse the information packed with model, and store
 * the parsed result into the params
 *
 * @param arg1 the information memory pointer
 * @param arg2 the size the information memory
 * @param arg3 the model name used for check whether the name in the information
 * @param arg4 the model configuration, ParseInfoFunc fill it with the
 * parsed information, the configuration will influence Network inference later
 *
 * @param arg5 the model IO information, ParseInfoFunc fill it with the parsed
 * information, the networkio will influence Network inference later
 *
 * @param arg6 the other configurations do not include in configuration and networkIO,
 * ParseInfoFunc fill it with the parsed information pair, now support:
 *
 * \verbatim
 * embed:rst:leading-asterisk
 *  .. list-table::
 *      :widths: 20 10 30
 *      :header-rows: 1
 *
 *      * - name
 *        - type
 *        - default
 *      * - "device_id"
 *        - int
 *        - 0
 *      * - "number_threads"
 *        - uint32_t
 *        - 1
 *      * - "is_inplace_model"
 *        - bool
 *        - false
 *      * - "use_tensorrt"
 *        - bool
 *        - false
 * \endverbatim
 *
 * @return Whether or not the parse function parse successfully
 */
using ParseInfoFunc = std::function<bool(
        const void*, size_t, const std::string model_name, Config& config,
        NetworkIO& network_io,
        std::unordered_map<std::string, LiteAny>& isolated_config_map,
        std::string& extra_info)>;

/**
 * @brief register a custom parser function to lite
 *
 * @param info_type  the name of the parser function, which will act as the
 * hash key to find the parser method.
 *
 * @param parse_func  the parser function, which will parse the given
 * information and modify the Network configuration and IO information.
 *
 * @return Whether or not the parse function register successful
 */
LITE_API bool register_parse_info_func(
        std::string info_type, const ParseInfoFunc& parse_func);

/** @brief get megengint lite version
 *
 * @param major  the major version of megengine lite
 * @param minor  the minor version of megengine lite
 * @param patch  the patch version of megengine lite
 */
LITE_API void get_version(int& major, int& minor, int& patch);

/**
 * @brief set the current log level
 * @param level the new log level to be set
 */
LITE_API void set_log_level(LiteLogLevel level);

/** @brief get the current log level
 * @return the current log level
 */
LITE_API LiteLogLevel get_log_level();

/** @brief get the number of device of the given device type in current context
 * @param device_type the to be count device type
 * @return the number of device
 */
LITE_API size_t get_device_count(LiteDeviceType device_type);

/** @brief try to coalesce all free memory in megenine, when call it MegEnine Lite
 * will try to free all the unused memory thus decrease the runtime memory usage
 */
LITE_API void try_coalesce_all_free_memory();

/**
 * @brief set the loader path to be used in lite
 * @param loader_path the file path which store the loader library
 */
LITE_API void set_loader_lib_path(const std::string& loader_path);

/**
 * @brief Set the algo policy cache file for CPU/CUDA, the algo policy cache is
 * produced by megengine fast-run
 *
 * @param cache_path  the file path which store the cache
 * @param always_sync  always update the cache file when model run
 */
LITE_API void set_persistent_cache(
        const std::string& cache_path, bool always_sync = false);

/**
 * @brief dump the PersistentCache policy cache to the specific file, if the network is
 * set to profile when forward, though this the algo policy will dump to file
 *
 * @param cache_path  the cache file path to be dump
 */
LITE_API void dump_persistent_cache(const std::string& cache_path);

/**
 * @brief set the TensorRT engine cache path for serialized prebuilt ICudaEngine
 *
 * @param cache_path  the cache file path to set
 */
LITE_API void set_tensor_rt_cache(std::string tensorrt_cache_path);

/**
 * @brief dump the TensorRT cache to the file set in set_tensor_rt_cache
 */
LITE_API void dump_tensor_rt_cache();

/**
 * @brief register the physical and virtual address pair to the mge, some device
 * need the map from physical to virtual
 *
 * @param vir_ptr - the virtual ptr to set to megenine
 * @param phy_ptr - the physical ptr to set to megenine
 * @param device - the device to set the pair memory
 * @param backend - the backend to set the pair memory
 *
 * @return Whether the register is successful
 */
LITE_API bool register_memory_pair(
        void* vir_ptr, void* phy_ptr, size_t length, LiteDeviceType device,
        LiteBackend backend = LiteBackend::LITE_DEFAULT);

/**
 * @brief clear the physical and virtual address pair in mge
 *
 * @param vir_ptr - the virtual ptr to set to megenine
 * @param phy_ptr - the physical ptr to set to megenine
 * @param device - the device to set the pair memory
 * @param backend - the backend to set the pair memory
 *
 * @return Whether the clear is successful
 */
LITE_API bool clear_memory_pair(
        void* vir_ptr, void* phy_ptr, LiteDeviceType device,
        LiteBackend backend = LiteBackend::LITE_DEFAULT);

/**
 * @brief get the physic address by the virtual address in mge.
 *
 * @param vir_ptr - the virtual ptr to set to megenine
 * @param device - the device to set the pair memory
 * @param backend - the backend to set the pair memory
 *
 * @return The physic address to lookup
 */
void* lookup_physic_ptr(void* vir_ptr, LiteDeviceType device, LiteBackend backend);

}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
