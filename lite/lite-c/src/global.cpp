#include "lite/global.h"
#include "common.h"
#include "lite-c/global_c.h"

namespace {
class ErrorMsg {
public:
    std::string& get_error_msg() { return error_msg; }
    ErrorCode get_error_code() { return error_code; }
    void set_error_msg(const std::string& msg, ErrorCode code) {
        error_msg = msg + ", Error Code: " + std::to_string(code);
        error_code = code;
    }

    void clear_error() {
        error_code = ErrorCode::OK;
        error_msg.clear();
    }

private:
    std::string error_msg;
    ErrorCode error_code;
};

static LITE_MUTEX mtx_error;
ErrorMsg& get_global_error() {
    static ErrorMsg error_msg;
    return error_msg;
}
}  // namespace

int LiteHandleException(const std::exception& e) {
    LITE_LOCK_GUARD(mtx_error);
    get_global_error().set_error_msg(e.what(), ErrorCode::LITE_INTERNAL_ERROR);
    return -1;
}

ErrorCode LITE_get_last_error_code() {
    LITE_LOCK_GUARD(mtx_error);
    return get_global_error().get_error_code();
}

void LITE_clear_last_error() {
    LITE_LOCK_GUARD(mtx_error);
    get_global_error().clear_error();
}

const char* LITE_get_last_error() {
    LITE_LOCK_GUARD(mtx_error);
    return get_global_error().get_error_msg().c_str();
}

int LITE_get_version(int* major, int* minor, int* patch) {
    LITE_ASSERT(major && minor && patch, "The ptr pass to LITE api is null");
    lite::get_version(*major, *minor, *patch);
    return 0;
}

int LITE_get_device_count(LiteDeviceType device_type, size_t* count) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(count, "The ptr pass to LITE api is null");
    *count = lite::get_device_count(device_type);
    LITE_CAPI_END();
}

int LITE_try_coalesce_all_free_memory() {
    LITE_CAPI_BEGIN();
    lite::try_coalesce_all_free_memory();
    LITE_CAPI_END();
}

int LITE_register_decryption_and_key(
        const char* decrypt_name, const LiteDecryptionFunc func,
        const uint8_t* key_data, size_t key_size) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(decrypt_name && key_data && func, "The ptr pass to LITE api is null");
    std::vector<uint8_t> key;
    for (size_t i = 0; i < key_size; i++) {
        key.push_back(key_data[i]);
    }
    auto decrypt_func = [func](const void* input_data, size_t input_size,
                               const std::vector<uint8_t>& key) {
        auto size = func(input_data, input_size, key.data(), key.size(), nullptr);
        std::vector<uint8_t> output(size, 0);
        func(input_data, input_size, key.data(), key.size(), output.data());
        return output;
    };
    lite::register_decryption_and_key(decrypt_name, decrypt_func, key);
    LITE_CAPI_END();
}

int LITE_update_decryption_or_key(
        const char* decrypt_name, const LiteDecryptionFunc func,
        const uint8_t* key_data, size_t key_size) {
    LITE_CAPI_BEGIN();
    std::vector<uint8_t> key;
    for (size_t i = 0; i < key_size; i++) {
        key.push_back(key_data[i]);
    }
    lite::DecryptionFunc decrypt_func = nullptr;
    if (func) {
        decrypt_func = [func](const void* input_data, size_t input_size,
                              const std::vector<uint8_t>& key) {
            auto size = func(input_data, input_size, key.data(), key.size(), nullptr);
            std::vector<uint8_t> output(size, 0);
            func(input_data, input_size, key.data(), key.size(), output.data());
            return output;
        };
    }
    lite::update_decryption_or_key(decrypt_name, decrypt_func, key);
    LITE_CAPI_END();
}

int LITE_register_parse_info_func(
        const char* info_type, const LiteParseInfoFunc parse_func) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(info_type && parse_func, "The ptr pass to LITE api is null");
    auto lite_func =
            [parse_func](
                    const void* info_data, size_t info_size,
                    const std::string model_name, lite::Config& config,
                    lite::NetworkIO& network_io,
                    std::unordered_map<std::string, lite::LiteAny>& separate_config_map,
                    std::string& extra_info) {
                LITE_MARK_USED_VAR(extra_info);
                size_t nr_threads = 1;
                int device_id = 0, is_cpu_inplace_mode = false, use_tensorrt = false;
                LiteNetworkIO c_io;
                LiteConfig c_config;
                auto ret = parse_func(
                        info_data, info_size, model_name.c_str(), &c_config, &c_io,
                        &device_id, &nr_threads, &is_cpu_inplace_mode, &use_tensorrt);
                config = convert_to_lite_config(c_config);
                network_io = convert_to_lite_io(c_io);
                if (device_id != 0) {
                    separate_config_map["device_id"] = device_id;
                }
                if (nr_threads != 1) {
                    separate_config_map["nr_threads"] =
                            static_cast<uint32_t>(nr_threads);
                }
                if (is_cpu_inplace_mode != false) {
                    separate_config_map["is_inplace_mode"] = is_cpu_inplace_mode;
                }
                if (use_tensorrt != false) {
                    separate_config_map["use_tensorrt"] = use_tensorrt;
                }
                return ret;
            };
    lite::register_parse_info_func(info_type, lite_func);
    LITE_CAPI_END();
}

int LITE_set_loader_lib_path(const char* loader_path) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(loader_path, "The ptr pass to LITE api is null");
    lite::set_loader_lib_path(loader_path);
    LITE_CAPI_END();
}

int LITE_set_persistent_cache(const char* cache_path, int always_sync) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(cache_path, "The ptr pass to LITE api is null");
    lite::set_persistent_cache(cache_path, always_sync);
    LITE_CAPI_END();
}

int LITE_set_tensor_rt_cache(const char* cache_path) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(cache_path, "The ptr pass to LITE api is null");
    lite::set_tensor_rt_cache(cache_path);
    LITE_CAPI_END();
}

int LITE_set_log_level(LiteLogLevel level) {
    LITE_CAPI_BEGIN();
    lite::set_log_level(level);
    LITE_CAPI_END();
}

int LITE_get_log_level(LiteLogLevel* level) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(level, "The ptr pass to LITE api is null");
    *level = lite::get_log_level();
    LITE_CAPI_END();
}

int LITE_dump_persistent_cache(const char* cache_path) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(cache_path, "The ptr pass to LITE api is null");
    lite::dump_persistent_cache(cache_path);
    LITE_CAPI_END();
}

int LITE_dump_tensor_rt_cache() {
    LITE_CAPI_BEGIN();
    lite::dump_tensor_rt_cache();
    LITE_CAPI_END();
}

int LITE_register_memory_pair(
        void* vir_ptr, void* phy_ptr, size_t length, LiteDeviceType device,
        LiteBackend backend) {
    LITE_CAPI_BEGIN();
    lite::register_memory_pair(vir_ptr, phy_ptr, length, device, backend);
    LITE_CAPI_END();
}

int LITE_clear_memory_pair(
        void* vir_ptr, void* phy_ptr, LiteDeviceType device, LiteBackend backend) {
    LITE_CAPI_BEGIN();
    lite::clear_memory_pair(vir_ptr, phy_ptr, device, backend);
    LITE_CAPI_END();
}

int LITE_lookup_physic_ptr(
        void* vir_ptr, void** phy_ptr, LiteDeviceType device, LiteBackend backend) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(vir_ptr && phy_ptr, "The ptr pass to vir and phy is nullptr");
    *phy_ptr = lite::lookup_physic_ptr(vir_ptr, device, backend);
    LITE_CAPI_END();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
