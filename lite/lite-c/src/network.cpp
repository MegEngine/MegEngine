/**
 * \file lite-c/src/network.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "lite/network.h"
#include "common.h"
#include "lite-c/network_c.h"

#include "../../src/network_impl_base.h"

#include <string.h>
#include <memory>
#include <mutex>
#include <unordered_map>

//! define a default Options
const LiteOptions default_option = {
        .weight_preprocess = false,
        .fuse_preprocess = false,
        .fake_next_exec = false,
        .var_sanity_check_first_run = true,
        .const_shape = false,
        .force_dynamic_alloc = false,
        .force_output_dynamic_alloc = false,
        .no_profiling_on_shape_change = false,
        .jit_level = 0,
        .comp_node_seq_record_level = 0,
        .graph_opt_level = 2,
        .async_exec_level = 1,
        //! layout transform options
        .enable_nchw44 = 0,
        .enable_nchw44_dot = 0,
        .enable_nchw88 = 0,
        .enable_nhwcd4 = 0,
        .enable_nchw4 = 0,
        .enable_nchw32 = 0,
        .enable_nchw64 = 0,

};

//! define a default config
LiteConfig default_config_t = {
        .has_compression = false,
        .device_id = -1,
        .device_type = LiteDeviceType::LITE_CPU,
        .backend = LiteBackend::LITE_DEFAULT,
        .bare_model_cryption_name = nullptr,
        .options = default_option};
LiteConfig* default_config() {
    return &default_config_t;
}

//! define a default IO
const LiteIO default_io = {
        .name = nullptr,
        .is_host = true,
        .io_type = LiteIOType::LITE_IO_VALUE,
        .config_layout = default_layout};

//! define a default NetworkIO
LiteNetworkIO default_network_io_t = {
        .inputs = nullptr, .outputs = nullptr, .input_size = 0, .output_size = 0};
LiteNetworkIO* default_network_io() {
    return &default_network_io_t;
}

namespace {
std::unordered_map<void*, std::shared_ptr<lite::Network>>& get_gloabl_network_holder() {
    static thread_local std::unordered_map<void*, std::shared_ptr<lite::Network>>
            network_holder;
    return network_holder;
}

/*!
 * \brief A user-implemented allocator interface
 */
class UserAllocator : public lite::Allocator {
public:
    UserAllocator(LiteAllocate allocate_func, LiteFree free_func)
            : m_allocator(allocate_func), m_free(free_func) {
        LITE_ASSERT(m_allocator && m_free);
    }

    //! allocate memory of size in the given device with the given align
    void* allocate(LiteDeviceType device_type, int device_id, size_t size, size_t align)
            override {
        return m_allocator(device_type, device_id, size, align);
    }

    //! free the memory pointed by ptr in the given device
    void free(LiteDeviceType device_type, int device_id, void* ptr) override {
        m_free(device_type, device_id, ptr);
    }

private:
    LiteAllocate m_allocator;
    LiteFree m_free;
};
}  // namespace

//! convert c config to lite::config
lite::Config convert_to_lite_config(const LiteConfig c_config) {
    lite::Config lite_config;
    lite_config.device_type = c_config.device_type;
    if (c_config.bare_model_cryption_name) {
        lite_config.bare_model_cryption_name = c_config.bare_model_cryption_name;
    }
    lite_config.backend = c_config.backend;
    lite_config.has_compression = c_config.has_compression;
    lite_config.device_id = c_config.device_id;

    lite_config.options.weight_preprocess = c_config.options.weight_preprocess;
    lite_config.options.fuse_preprocess = c_config.options.fuse_preprocess;
    lite_config.options.fake_next_exec = c_config.options.fake_next_exec;
    lite_config.options.var_sanity_check_first_run =
            c_config.options.var_sanity_check_first_run;
    lite_config.options.const_shape = c_config.options.const_shape;
    lite_config.options.force_dynamic_alloc = c_config.options.const_shape;
    lite_config.options.force_output_dynamic_alloc =
            c_config.options.force_output_dynamic_alloc;
    lite_config.options.no_profiling_on_shape_change =
            c_config.options.no_profiling_on_shape_change;
    lite_config.options.jit_level = c_config.options.jit_level;
    lite_config.options.comp_node_seq_record_level =
            c_config.options.comp_node_seq_record_level;
    lite_config.options.graph_opt_level = c_config.options.graph_opt_level;
    lite_config.options.async_exec_level = c_config.options.async_exec_level;

    lite_config.options.enable_nchw44 = c_config.options.enable_nchw44;
    lite_config.options.enable_nchw44_dot = c_config.options.enable_nchw44_dot;
    lite_config.options.enable_nchw88 = c_config.options.enable_nchw88;
    lite_config.options.enable_nchw4 = c_config.options.enable_nchw4;
    lite_config.options.enable_nhwcd4 = c_config.options.enable_nhwcd4;
    lite_config.options.enable_nchw32 = c_config.options.enable_nchw32;
    lite_config.options.enable_nchw64 = c_config.options.enable_nchw64;

    return lite_config;
}

//! convert C NetworkIO io to lite::NetworkIO
lite::NetworkIO convert_to_lite_io(const LiteNetworkIO c_network_io) {
    lite::NetworkIO network_io;
    for (size_t i = 0; i < c_network_io.input_size; i++) {
        LiteIO* c_io = c_network_io.inputs + i;
        LITE_ASSERT(c_io->name, "input name of io tensor must set.");
        network_io.inputs.push_back(
                {c_io->name, static_cast<bool>(c_io->is_host), c_io->io_type,
                 convert_to_layout(c_io->config_layout)});
    }
    for (size_t i = 0; i < c_network_io.output_size; i++) {
        LiteIO* c_io = c_network_io.outputs + i;
        LITE_ASSERT(c_io->name, "output name of io tensor must set.");
        network_io.outputs.push_back(
                {c_io->name, static_cast<bool>(c_io->is_host), c_io->io_type,
                 convert_to_layout(c_io->config_layout)});
    }
    return network_io;
}

int LITE_make_default_network(LiteNetwork* network) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    auto lite_network = std::make_shared<lite::Network>();
    get_gloabl_network_holder()[lite_network.get()] = lite_network;
    *network = lite_network.get();
    LITE_CAPI_END();
}

int LITE_make_network(
        LiteNetwork* network, const LiteConfig config, const LiteNetworkIO network_io) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    auto lite_network = std::make_shared<lite::Network>(
            convert_to_lite_config(config), convert_to_lite_io(network_io));
    get_gloabl_network_holder()[lite_network.get()] = lite_network;
    *network = lite_network.get();
    LITE_CAPI_END();
}

int LITE_make_network_config(LiteNetwork* network, const LiteConfig config) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    auto lite_network = std::make_shared<lite::Network>(convert_to_lite_config(config));
    get_gloabl_network_holder()[lite_network.get()] = lite_network;
    *network = lite_network.get();
    LITE_CAPI_END();
}

int LITE_load_model_from_mem(LiteNetwork network, void* model_mem, size_t size) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    LITE_ASSERT(model_mem, "The model memory pass to LITE api is null");
    static_cast<lite::Network*>(network)->load_model(model_mem, size);
    LITE_CAPI_END();
}

int LITE_load_model_from_path(LiteNetwork network, const char* model_path) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    LITE_ASSERT(model_path, "The model path pass to LITE api is null");
    static_cast<lite::Network*>(network)->load_model(model_path);
    LITE_CAPI_END();
}

int LITE_destroy_network(LiteNetwork network) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    get_gloabl_network_holder().erase(network);
    LITE_CAPI_END();
}

int LITE_forward(const LiteNetwork network) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    static_cast<lite::Network*>(network)->forward();
    LITE_CAPI_END();
}

int LITE_wait(const LiteNetwork network) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    static_cast<lite::Network*>(network)->wait();
    LITE_CAPI_END();
}

int LITE_get_io_tensor(
        LiteNetwork network, const char* io_name, LiteTensorPhase phase,
        LiteTensor* tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    auto io_tensor =
            static_cast<lite::Network*>(network)->get_io_tensor(io_name, phase);
    *tensor = io_tensor.get();
    LITE_CAPI_END();
}

int LITE_get_input_name(const LiteNetwork network, size_t index, const char** name) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network && name, "The network pass to LITE api is null");
    *name = lite::NetworkHelper::implement(static_cast<lite::Network*>(network))
                    ->get_input_name(index);
    LITE_CAPI_END();
}

int LITE_get_output_name(const LiteNetwork network, size_t index, const char** name) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    LITE_ASSERT(name, "The name ptr pass to LITE api is null");
    *name = lite::NetworkHelper::implement(static_cast<lite::Network*>(network))
                    ->get_output_name(index);
    LITE_CAPI_END();
}

int LITE_get_all_input_name(
        const LiteNetwork network, size_t* size, const char** name) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    auto&& names = lite::NetworkHelper::implement(static_cast<lite::Network*>(network))
                           ->get_all_input_name();
    if (size)
        *size = names.size();
    if (name) {
        for (auto in_name : names) {
            *name = in_name;
            name++;
        }
    }
    LITE_CAPI_END();
}

int LITE_get_all_output_name(
        const LiteNetwork network, size_t* size, const char** name) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    auto&& names = lite::NetworkHelper::implement(static_cast<lite::Network*>(network))
                           ->get_all_output_name();
    if (size)
        *size = names.size();
    if (name) {
        for (auto in_name : names) {
            *name = in_name;
            name++;
        }
    }
    LITE_CAPI_END();
}

int LITE_set_device_id(LiteNetwork network, int device_id) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    static_cast<lite::Network*>(network)->set_device_id(device_id);
    LITE_CAPI_END();
}

int LITE_get_device_id(const LiteNetwork network, int* device_id) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    LITE_ASSERT(device_id, "The device_id pass to LITE api is null");
    *device_id = static_cast<lite::Network*>(network)->get_device_id();
    LITE_CAPI_END();
}

int LITE_set_stream_id(LiteNetwork network, int stream_id) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    static_cast<lite::Network*>(network)->set_stream_id(stream_id);
    LITE_CAPI_END();
}

int LITE_get_stream_id(const LiteNetwork network, int* stream_id) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    LITE_ASSERT(stream_id, "The stream_id pass to LITE api is null");
    *stream_id = static_cast<lite::Network*>(network)->get_stream_id();
    LITE_CAPI_END();
}

int LITE_get_model_extra_info(
        const LiteNetwork network, const char** info, int* info_size) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    LITE_ASSERT(info_size, "The info and info_size are all null");
    auto& extra_info = static_cast<lite::Network*>(network)->get_model_extra_info();
    *info_size = extra_info.size();
    *info = extra_info.c_str();
    LITE_MARK_USED_VAR(info);
    LITE_CAPI_END();
}

int LITE_get_device_type(const LiteNetwork network, LiteDeviceType* device_type) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    LITE_ASSERT(device_type, "The device_type pass to LITE api is null");
    *device_type = static_cast<lite::Network*>(network)->get_device_type();
    LITE_CAPI_END();
}

int LITE_set_async_callback(
        LiteNetwork network, const LiteAsyncCallback async_callback) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    LITE_ASSERT(async_callback, "The ptr pass to LITE api is null");
    static_cast<lite::Network*>(network)->set_async_callback(std::move(async_callback));
    LITE_CAPI_END();
}

int LITE_set_start_callback(
        LiteNetwork network, const LiteStartCallback start_callback) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    auto lite_start_callback =
            [start_callback](const std::unordered_map<
                             std::string,
                             std::pair<lite::IO, std::shared_ptr<lite::Tensor>>>&
                                     inputs_map) -> void {
        std::vector<LiteIO> ios;
        std::vector<LiteTensor> io_tensors;
        size_t nr_io = 0;
        for (const auto& io : inputs_map) {
            nr_io++;
            auto&& lite_io = io.second.first;
            ios.push_back(
                    {lite_io.name.c_str(), lite_io.is_host, lite_io.io_type,
                     convert_to_clayout(lite_io.config_layout)});
            io_tensors.push_back(io.second.second.get());
        }
        start_callback(ios.data(), io_tensors.data(), nr_io);
    };
    static_cast<lite::Network*>(network)->set_start_callback(lite_start_callback);
    LITE_CAPI_END();
}

int LITE_set_finish_callback(
        LiteNetwork network, const LiteFinishCallback finish_callback) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    auto lite_finish_callback =
            [finish_callback](const std::unordered_map<
                              std::string,
                              std::pair<lite::IO, std::shared_ptr<lite::Tensor>>>&
                                      outputs_map) -> void {
        std::vector<LiteIO> ios;
        std::vector<LiteTensor> io_tensors;
        size_t nr_io = 0;
        for (const auto& io : outputs_map) {
            nr_io++;
            auto&& lite_io = io.second.first;
            ios.push_back(
                    {lite_io.name.c_str(), lite_io.is_host, lite_io.io_type,
                     convert_to_clayout(lite_io.config_layout)});
            io_tensors.push_back(io.second.second.get());
        }
        finish_callback(ios.data(), io_tensors.data(), nr_io);
    };
    static_cast<lite::Network*>(network)->set_finish_callback(lite_finish_callback);
    LITE_CAPI_END();
}

int LITE_enable_profile_performance(
        LiteNetwork network, const char* profile_json_file_path) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    static_cast<lite::Network*>(network)->enable_profile_performance(
            profile_json_file_path);
    LITE_CAPI_END();
}

int LITE_is_cpu_inplace_mode(const LiteNetwork network, int* is_cpu_inplace_mode) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network && is_cpu_inplace_mode, "The network pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    *is_cpu_inplace_mode = lite::Runtime::is_cpu_inplace_mode(network_shared);
    LITE_CAPI_END();
}

int LITE_get_cpu_threads_number(const LiteNetwork network, size_t* nr_threads) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    LITE_ASSERT(nr_threads, "The ptr pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    *nr_threads = lite::Runtime::get_cpu_threads_number(network_shared);
    LITE_CAPI_END();
}

int LITE_set_cpu_inplace_mode(LiteNetwork network) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    lite::Runtime::set_cpu_inplace_mode(network_shared);
    LITE_CAPI_END();
}

int LITE_use_tensorrt(LiteNetwork network) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    lite::Runtime::use_tensorrt(network_shared);
    LITE_CAPI_END();
}

int LITE_set_cpu_threads_number(LiteNetwork network, size_t nr_threads) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    lite::Runtime::set_cpu_threads_number(network_shared, nr_threads);
    LITE_CAPI_END();
}

int LITE_set_network_algo_policy(LiteNetwork network, LiteAlgoSelectStrategy strategy) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    lite::Runtime::set_network_algo_policy(network_shared, strategy);
    LITE_CAPI_END();
}

int LITE_set_network_algo_fastrun_config(
        LiteNetwork network, unsigned int shared_batch_size,
        int binary_equal_between_batch) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    lite::Runtime::set_network_algo_policy(
            network_shared, LiteAlgoSelectStrategy(0), shared_batch_size,
            binary_equal_between_batch);
    LITE_CAPI_END();
}

int LITE_set_network_algo_workspace_limit(LiteNetwork network, size_t workspace_limit) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    lite::Runtime::set_network_algo_workspace_limit(network_shared, workspace_limit);
    LITE_CAPI_END();
}

int LITE_set_runtime_thread_affinity(
        LiteNetwork network,
        const LiteThreadAffinityCallback thread_affinity_callback) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    lite::Runtime::set_runtime_thread_affinity(
            network_shared, std::move(thread_affinity_callback));
    LITE_CAPI_END();
}

int LITE_set_memory_allocator(
        LiteNetwork network, const LiteAllocate allocate_fun, const LiteFree free_fun) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(
            network && allocate_fun && free_fun, "The ptr pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    lite::Runtime::set_memory_allocator(
            network_shared, std::make_shared<UserAllocator>(allocate_fun, free_fun));
    LITE_CAPI_END();
}

int LITE_enable_io_txt_dump(LiteNetwork network, const char* io_txt_out_file) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    lite::Runtime::enable_io_txt_dump(network_shared, io_txt_out_file);
    LITE_CAPI_END();
}

int LITE_enable_io_bin_dump(LiteNetwork network, const char* io_bin_out_dir) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(network, "The network pass to LITE api is null");
    std::shared_ptr<lite::Network> network_shared{
            static_cast<lite::Network*>(network), [](void*) {}};
    lite::Runtime::enable_io_bin_dump(network_shared, io_bin_out_dir);
    LITE_CAPI_END();
}

int LITE_shared_weight_with_network(
        LiteNetwork dst_network, const LiteNetwork src_network) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(dst_network && src_network, "The network pass to LITE api is null");
    const std::shared_ptr<lite::Network> src_shared_net{
            static_cast<lite::Network*>(src_network), [](void*) {}};
    std::shared_ptr<lite::Network> dst_shared_net{
            static_cast<lite::Network*>(dst_network), [](void*) {}};
    lite::Runtime::shared_weight_with_network(dst_shared_net, src_shared_net);
    LITE_CAPI_END();
}

int LITE_share_runtime_memroy(LiteNetwork dst_network, LiteNetwork src_network) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(src_network && dst_network, "The network pass to LITE api is null");
    std::shared_ptr<lite::Network> src_shared{
            static_cast<lite::Network*>(src_network), [](void*) {}};
    std::shared_ptr<lite::Network> dst_shared{
            static_cast<lite::Network*>(dst_network), [](void*) {}};
    lite::Runtime::share_runtime_memory_with(dst_shared, src_shared);
    LITE_CAPI_END();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
