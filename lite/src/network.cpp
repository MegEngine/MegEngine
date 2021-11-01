/**
 * \file src/network.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "lite/network.h"
#include "function_base.h"
#include "network_impl_base.h"
#include "parse_info/parse_info_base.h"
#include "parse_model/model_parser.h"
#include "type_info.h"
#if LITE_BUILD_WITH_MGE
#include "mge/function_dft.h"
#include "mge/network_impl.h"
#endif

#include <fstream>
#include <memory>

using namespace lite;

/**
 * \brief Construct the new work implement
 * the order must be :
 * 1. creeat the implement
 * 2. config and load
 * 3. set_io
 */
Network::Network(const Config& config, const NetworkIO& network_io) {
    LITE_ERROR_HANDLER_BEGIN
    m_config = config;
    m_network_io = network_io;
    if (config.backend == LiteBackend::LITE_DEFAULT) {
        m_impl = call_func<
                NetworkImplDft, std::unique_ptr<lite::Network::NetworkImplBase>>(
                "create_network");
    }
    m_impl->set_config(config);
    m_impl->set_io(network_io);
    LITE_ERROR_HANDLER_END
}

Network::Network(const NetworkIO& network_io, const Config& config) {
    LITE_ERROR_HANDLER_BEGIN
    m_config = config;
    m_network_io = network_io;
    if (config.backend == LiteBackend::LITE_DEFAULT) {
        m_impl = call_func<
                NetworkImplDft, std::unique_ptr<lite::Network::NetworkImplBase>>(
                "create_network");
    }
    m_impl->set_config(config);
    m_impl->set_io(network_io);
    LITE_ERROR_HANDLER_END
}

void Network::load_model(void* model_mem, size_t size) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    //! this model_mem is managed by user
    std::shared_ptr<void> model{model_mem, [](void*) {}};
    prase_model(model, size);
    LITE_ERROR_HANDLER_END
}

void Network::load_model(std::string model_path) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    FILE* fin = fopen(model_path.c_str(), "rb");
    LITE_ASSERT(fin, "failed to open %s: %s", model_path.c_str(), strerror(errno));
    fseek(fin, 0, SEEK_END);
    size_t size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    void* ptr = malloc(size);
    std::shared_ptr<void> buf{ptr, ::free};
    auto nr = fread(buf.get(), 1, size, fin);
    LITE_ASSERT(nr == size);
    fclose(fin);
    prase_model(buf, size);
    LITE_ERROR_HANDLER_END
}

void Network::prase_model(std::shared_ptr<void> model_data, size_t size) {
    std::unordered_map<std::string, LiteAny> separate_config_map;
    ModelParser model_parser(model_data, size);
    //! parse the model info
    if (model_parser.parse_model_info(
                m_config, m_network_io, separate_config_map, m_extra_info)) {
        if (m_config.backend == LiteBackend::LITE_DEFAULT &&
            m_impl->get_backend_type() != LiteBackend::LITE_DEFAULT) {
            m_impl.reset(try_call_func<NetworkImplDft, lite::Network::NetworkImplBase*>(
                    "parse_model"));
        }
        m_impl->set_config(m_config);
        m_impl->set_io(m_network_io);
    }
    //! decryption the model
    size_t model_length;
    auto&& model_shared_ptr = model_parser.parse_model(model_length, m_config);

    m_impl->load_model(model_shared_ptr, model_length, separate_config_map);
    m_loaded = true;
    update_from_implement();
}

Network::~Network() = default;

void Network::update_from_implement() {
    m_config.device_type = m_impl->get_device_type();
}

void Network::compute_only_configured_output() {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(
            !m_loaded,
            "compute_only_configured_output should be used before model "
            "loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    return m_impl->compute_only_configured_output();
    LITE_ERROR_HANDLER_END
}

std::shared_ptr<Tensor> Network::get_io_tensor(
        std::string name, LiteTensorPhase phase) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(m_loaded, "get_io_tensor should be used after model loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    return m_impl->get_io_tensor(name, phase);
    LITE_ERROR_HANDLER_END
}

std::shared_ptr<Tensor> Network::get_input_tensor(size_t index) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(m_loaded, "get_input_tensor should be used after model loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    return m_impl->get_input_tensor(index);
    LITE_ERROR_HANDLER_END
}

std::shared_ptr<Tensor> Network::get_output_tensor(size_t index) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(m_loaded, "get_output_tensor should be used after model loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    return m_impl->get_output_tensor(index);
    LITE_ERROR_HANDLER_END
}

Network& Network::set_async_callback(const AsyncCallback& callback) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(
            !m_config.options.force_output_use_user_specified_memory,
            "Async mode can't run with force_output_use_user_specified_memory which "
            "output data is written to use specific memory.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    m_impl->set_async_callback(std::move(callback));
    return *this;
    LITE_ERROR_HANDLER_END
}

Network& Network::set_start_callback(const StartCallback& callback) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    m_impl->set_start_callback(std::move(callback));
    return *this;
    LITE_ERROR_HANDLER_END
}

Network& Network::set_finish_callback(const FinishCallback& callback) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    m_impl->set_finish_callback(std::move(callback));
    return *this;
    LITE_ERROR_HANDLER_END
}

Network& Network::set_device_id(int device_id) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(!m_loaded, "set_device_id should be used before model loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    m_impl->set_device_id(device_id);
    return *this;
    LITE_ERROR_HANDLER_END
}

Network& Network::set_stream_id(int stream_id) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(!m_loaded, "set_stream_id should be used before model loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    m_impl->set_stream_id(stream_id);
    return *this;
    LITE_ERROR_HANDLER_END
}

void Network::forward() {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(m_loaded, "forward should be used after model loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl.get());
    m_impl->forward();
    LITE_ERROR_HANDLER_END
}

void Network::wait() {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(m_loaded, "wait should be used after model loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    m_impl->wait();
    LITE_ERROR_HANDLER_END
}

std::string Network::get_input_name(size_t index) const {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(m_loaded, "get_input_name should be used after model loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    return m_impl->get_input_name(index);
    LITE_ERROR_HANDLER_END
}

std::string Network::get_output_name(size_t index) const {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(m_loaded, "get_output_name should be used after model loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    return m_impl->get_output_name(index);
    LITE_ERROR_HANDLER_END
}

std::vector<std::string> Network::get_all_input_name() const {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(m_loaded, "get_all_input_name should be used after model loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    auto all_input_name = m_impl->get_all_input_name();
    std::vector<std::string> all_names;
    for (auto& name : all_input_name) {
        all_names.push_back(name);
    }
    return all_names;
    LITE_ERROR_HANDLER_END
}

std::vector<std::string> Network::get_all_output_name() const {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(m_loaded, "get_all_output_name should be used after model loaded.");
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    auto all_output_name = m_impl->get_all_output_name();
    std::vector<std::string> all_names;
    for (auto& name : all_output_name) {
        all_names.push_back(name);
    }
    return all_names;
    LITE_ERROR_HANDLER_END
}

int Network::get_device_id() const {
    LITE_ERROR_HANDLER_BEGIN
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    return m_impl->get_device_id();
    LITE_ERROR_HANDLER_END
}

int Network::get_stream_id() const {
    LITE_ERROR_HANDLER_BEGIN
    LITE_CHECK_NON_NULL_POINTER(m_impl);
    return m_impl->get_stream_id();
    LITE_ERROR_HANDLER_END
}

void Network::enable_profile_performance(std::string profile_file_path) {
    LITE_ERROR_HANDLER_BEGIN
    m_impl->enable_profile_performance(profile_file_path);
    LITE_ERROR_HANDLER_END
}

const std::string& Network::get_model_extra_info() {
    LITE_ERROR_HANDLER_BEGIN
    return m_extra_info;
    LITE_ERROR_HANDLER_END
}

LiteDeviceType Network::get_device_type() const {
    LITE_ERROR_HANDLER_BEGIN
    return m_impl->get_device_type();
    LITE_ERROR_HANDLER_END
}

void Network::get_static_memory_alloc_info(const std::string& log_dir) const {
    LITE_ERROR_HANDLER_BEGIN
#ifndef __IN_TEE_ENV__
#if MGB_ENABLE_JSON
    LITE_ASSERT(m_loaded, "get_all_output_name should be used after model loaded.");
    m_impl->get_static_memory_alloc_info(log_dir);
    return;
#endif
#endif
    LITE_MARK_USED_VAR(log_dir);
    LITE_THROW("Doesn't support get_static_memory_alloc_info().Please check macro.");
    LITE_ERROR_HANDLER_END
}

/*********************** MGE special network function ***************/

void Runtime::set_cpu_threads_number(
        std::shared_ptr<Network> network, size_t nr_threads) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl = NetworkHelper::implement(network);
    if (network_impl->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        LITE_ASSERT(
                !NetworkHelper::loaded(network),
                "set_cpu_threads_number should be used before model loaded.");
        call_func<NetworkImplDft, void>(
                "set_cpu_threads_number", network_impl, nr_threads);
        return;
    }
    LITE_THROW("set_cpu_threads_number is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

void Runtime::use_tensorrt(std::shared_ptr<Network> network) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl = NetworkHelper::implement(network);
    if (network_impl->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        LITE_ASSERT(
                !NetworkHelper::loaded(network),
                "use_tensorrt should be used before model loaded.");
        call_func<NetworkImplDft, void>("use_tensorrt", network_impl);
        return;
    }
    LITE_THROW("use_tensorrt is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

size_t Runtime::get_cpu_threads_number(const std::shared_ptr<Network> network) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl = NetworkHelper::implement(network);
    if (network_impl->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        return call_func<NetworkImplDft, size_t>(
                "get_cpu_threads_number", network_impl);
    }
    LITE_THROW("get_cpu_threads_number is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

void Runtime::set_runtime_thread_affinity(
        std::shared_ptr<Network> network,
        const ThreadAffinityCallback& thread_affinity_callback) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl = NetworkHelper::implement(network);
    if (network_impl->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        LITE_ASSERT(
                NetworkHelper::loaded(network),
                "set_runtime_thread_affinity should be used after model "
                "loaded.");
        call_func<NetworkImplDft, void>(
                "set_runtime_thread_affinity", network_impl, thread_affinity_callback);

        return;
    }
    LITE_THROW("set_runtime_thread_affinity is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

void Runtime::set_cpu_inplace_mode(std::shared_ptr<Network> network) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl = NetworkHelper::implement(network);
    if (network_impl->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        LITE_ASSERT(
                !NetworkHelper::loaded(network),
                "set_cpu_inplace_mode should be used before model loaded.");
        call_func<NetworkImplDft, void>("set_cpu_inplace_mode", network_impl);
        return;
    }
    LITE_THROW("set_cpu_inplace_mode is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

bool Runtime::is_cpu_inplace_mode(const std::shared_ptr<Network> network) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl = NetworkHelper::implement(network);
    if (network_impl->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        return call_func<NetworkImplDft, bool>("is_cpu_inplace_mode", network_impl);
    }
    LITE_THROW("is_cpu_inplace_mode is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

//! set opr algorithm selection strategy in the network
void Runtime::set_network_algo_policy(
        std::shared_ptr<Network> network, LiteAlgoSelectStrategy strategy,
        uint32_t shared_batch_size, bool binary_equal_between_batch) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl = NetworkHelper::implement(network);
    if (network_impl->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        call_func<NetworkImplDft, void>(
                "set_network_algo_policy", network_impl, strategy, shared_batch_size,
                binary_equal_between_batch);
        return;
    }
    LITE_THROW("set_network_algo_policy is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

//! set opr algorithm selection strategy in the network
void Runtime::set_network_algo_workspace_limit(
        std::shared_ptr<Network> network, size_t workspace_limit) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl = NetworkHelper::implement(network);
    if (network_impl->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        LITE_ASSERT(
                NetworkHelper::loaded(network),
                "set_network_algo_policy should be used after model "
                "loaded.");
        call_func<NetworkImplDft, void>(
                "set_network_algo_workspace_limit", network_impl, workspace_limit);
        return;
    }
    LITE_THROW(
            "set_network_algo_workspace_limit is not aviliable in the "
            "backend.");
    LITE_ERROR_HANDLER_END
}

//! set the network memroy allocator, the allocator is defined by user
void Runtime::set_memory_allocator(
        std::shared_ptr<Network> network, std::shared_ptr<Allocator> user_allocator) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl = NetworkHelper::implement(network);
    if (network_impl->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        LITE_ASSERT(
                !NetworkHelper::loaded(network),
                "set_memory_allocator should be used before model loaded.");
        call_func<NetworkImplDft, void>(
                "set_memory_allocator", network_impl, user_allocator);
        return;
    }
    LITE_THROW("set_memory_allocator is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

void Runtime::share_runtime_memory_with(
        std::shared_ptr<Network> dst_network, std::shared_ptr<Network> src_network) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl_dst = NetworkHelper::implement(dst_network);
    if (network_impl_dst->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        LITE_ASSERT(
                !NetworkHelper::loaded(dst_network),
                "share_runtime_memory_with should be used before model "
                "loaded.");
        call_func<NetworkImplDft, void>(
                "share_runtime_memory_with", network_impl_dst,
                NetworkHelper::implement(src_network));
        return;
    }
    LITE_THROW("share_runtime_memory_with is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

void Runtime::enable_io_txt_dump(
        std::shared_ptr<Network> network, std::string io_txt_out_file) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl = NetworkHelper::implement(network);
    if (network_impl->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        call_func<NetworkImplDft, void>(
                "enable_io_txt_dump", network_impl, io_txt_out_file);
        return;
    }
    LITE_THROW("enable_io_txt_dump is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

void Runtime::enable_io_bin_dump(
        std::shared_ptr<Network> network, std::string io_bin_out_dir) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl = NetworkHelper::implement(network);
    if (network_impl->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        call_func<NetworkImplDft, void>(
                "enable_io_bin_dump", network_impl, io_bin_out_dir);
        return;
    }
    LITE_THROW("enable_io_bin_dump is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

void Runtime::shared_weight_with_network(
        std::shared_ptr<Network> dst_network,
        const std::shared_ptr<Network> src_network) {
    LITE_ERROR_HANDLER_BEGIN
    auto network_impl_dst = NetworkHelper::implement(dst_network);
    if (network_impl_dst->get_backend_type() == LiteBackend::LITE_DEFAULT) {
        LITE_ASSERT(
                NetworkHelper::loaded(src_network),
                "shared_weight_with_network should be used after the src "
                "network "
                "loaded.");
        auto src_implment = NetworkHelper::implement(src_network);
        call_func<NetworkImplDft, void>(
                "shared_weight_with", network_impl_dst, src_implment);
        NetworkHelper::loaded(dst_network, true);
        return;
    }
    LITE_THROW("shared_weight_with_network is not aviliable in the backend.");
    LITE_ERROR_HANDLER_END
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
