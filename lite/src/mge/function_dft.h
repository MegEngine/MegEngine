
/**
 * \file src/mge/function_dft.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#if LITE_BUILD_WITH_MGE
#include "function_base.h"
#include "network_impl.h"
#include "network_impl_base.h"
#include "tensor_impl.h"
namespace lite {

#define THROW_FUNC_ERROR(func_name)                                   \
    auto msg_info = func_name + "  is not aviliable in Dft backend."; \
    LITE_THROW(msg_info.c_str())

// the functions used for dft's tensor.cpp are as followed:

template <>
inline std::shared_ptr<Tensor::TensorImplBase> call_func<
        TensorImplDft, std::shared_ptr<Tensor::TensorImplBase>>(std::string func_name) {
    if (func_name == "create_tensor") {
        return std::make_shared<TensorImplDft>();
    }
    THROW_FUNC_ERROR(func_name);
}

template <>
inline std::shared_ptr<Tensor::TensorImplBase> call_func<
        TensorImplDft, std::shared_ptr<Tensor::TensorImplBase>>(
        std::string func_name, LiteDeviceType device_type, bool is_pinned_host) {
    if (func_name == "create_tensor") {
        return std::make_shared<TensorImplDft>(device_type, is_pinned_host);
    }
    THROW_FUNC_ERROR(func_name);
}

template <>
inline std::shared_ptr<Tensor::TensorImplBase> call_func<
        TensorImplDft, std::shared_ptr<Tensor::TensorImplBase>>(
        std::string func_name, int device_id, LiteDeviceType device_type,
        const Layout layout, bool is_pinned_host) {
    if (func_name == "create_tensor") {
        return std::make_shared<TensorImplDft>(
                device_id, device_type, layout, is_pinned_host);
    }
    THROW_FUNC_ERROR(func_name);
}

template <>
inline std::shared_ptr<Tensor::TensorImplBase> call_func<
        TensorImplDft, std::shared_ptr<Tensor::TensorImplBase>>(
        std::string func_name, LiteDeviceType device_type, const Layout layout,
        bool is_pinned_host) {
    if (func_name == "create_tensor") {
        return std::make_shared<TensorImplDft>(device_type, layout, is_pinned_host);
    }
    THROW_FUNC_ERROR(func_name);
}

template <>
inline std::shared_ptr<Tensor::TensorImplBase> call_func<
        TensorImplDft, std::shared_ptr<Tensor::TensorImplBase>>(
        std::string func_name, int device_id, int stream_id, LiteDeviceType device_type,
        bool is_pinned_host) {
    if (func_name == "create_tensor") {
        return std::make_shared<TensorImplDft>(
                device_id, stream_id, device_type, is_pinned_host);
    }
    THROW_FUNC_ERROR(func_name);
}

// the functions used for dft's network.cpp are as followed:

template <>
inline std::unique_ptr<Network::NetworkImplBase> call_func<
        NetworkImplDft, std::unique_ptr<Network::NetworkImplBase>>(
        std::string func_name) {
    if (func_name == "create_network") {
        return std::make_unique<NetworkImplDft>();
    }
    THROW_FUNC_ERROR(func_name);
}

template <>
inline Network::NetworkImplBase* try_call_func<
        NetworkImplDft, Network::NetworkImplBase*>(std::string func_name) {
    if (func_name == "parse_model") {
        return new NetworkImplDft();
    }
    THROW_FUNC_ERROR(func_name);
}

#define CALL_FUNC(func_name, ...) \
    network_impl->cast_final_safe<NetworkImplDft>().func_name(__VA_ARGS__)

template <>
inline void call_func<NetworkImplDft, void>(
        std::string func_name, Network::NetworkImplBase* network_impl, size_t num) {
    if (func_name == "set_cpu_threads_number") {
        CALL_FUNC(set_cpu_threads_number, num);
    } else if (func_name == "set_network_algo_workspace_limit") {
        CALL_FUNC(set_network_algo_workspace_limit, num);
    } else {
        THROW_FUNC_ERROR(func_name);
    }
}

template <>
inline void call_func<NetworkImplDft, void>(
        std::string func_name, Network::NetworkImplBase* network_impl) {
    if (func_name == "use_tensorrt") {
        CALL_FUNC(use_tensorrt);
    } else if (func_name == "set_cpu_inplace_mode") {
        CALL_FUNC(set_cpu_inplace_mode);
    } else if (func_name == "enable_global_layout_transform") {
        CALL_FUNC(enable_global_layout_transform);
    } else {
        THROW_FUNC_ERROR(func_name);
    }
}

template <>
inline size_t call_func<NetworkImplDft, size_t>(
        std::string func_name, Network::NetworkImplBase* network_impl) {
    if (func_name == "get_cpu_threads_number") {
        return CALL_FUNC(get_cpu_threads_number);
    }
    THROW_FUNC_ERROR(func_name);
}

template <>
inline bool call_func<NetworkImplDft, bool>(
        std::string func_name, Network::NetworkImplBase* network_impl) {
    if (func_name == "is_cpu_inplace_mode") {
        return CALL_FUNC(is_cpu_inplace_mode);
    }
    THROW_FUNC_ERROR(func_name);
}

template <>
inline void call_func<NetworkImplDft, void>(
        std::string func_name, Network::NetworkImplBase* network_impl,
        ThreadAffinityCallback thread_affinity_callback) {
    if (func_name == "set_runtime_thread_affinity") {
        return CALL_FUNC(
                set_runtime_thread_affinity, std::move(thread_affinity_callback));
    }
    THROW_FUNC_ERROR(func_name);
}

template <>
inline void call_func<NetworkImplDft, void>(
        std::string func_name, Network::NetworkImplBase* network_impl,
        LiteAlgoSelectStrategy strategy, uint32_t shared_batch_size,
        bool binary_equal_between_batch) {
    if (func_name == "set_network_algo_policy") {
        return CALL_FUNC(
                set_network_algo_policy, strategy, shared_batch_size,
                binary_equal_between_batch);
    }
    THROW_FUNC_ERROR(func_name);
}

template <>
inline void call_func<NetworkImplDft, void>(
        std::string func_name, Network::NetworkImplBase* network_impl,
        std::shared_ptr<Allocator> user_allocator) {
    if (func_name == "set_memory_allocator") {
        return CALL_FUNC(set_memory_allocator, user_allocator);
    }
    THROW_FUNC_ERROR(func_name);
}

template <>
inline void call_func<NetworkImplDft, void>(
        std::string func_name, Network::NetworkImplBase* network_impl,
        std::string file_name) {
    if (func_name == "enable_io_txt_dump") {
        return CALL_FUNC(enable_io_txt_dump, file_name);
    } else if (func_name == "enable_io_bin_dump") {
        return CALL_FUNC(enable_io_bin_dump, file_name);
    } else if (func_name == "dump_layout_transform_model") {
        return CALL_FUNC(dump_layout_transform_model, file_name);
    }
    THROW_FUNC_ERROR(func_name);
}

template <>
inline void call_func<NetworkImplDft, void>(
        std::string func_name, Network::NetworkImplBase* network_impl,
        Network::NetworkImplBase* src_network_impl) {
    if (func_name == "share_runtime_memory_with") {
        CALL_FUNC(share_runtime_memory_with, src_network_impl);
    } else if (func_name == "shared_weight_with") {
        CALL_FUNC(shared_weight_with, src_network_impl);
    } else {
        THROW_FUNC_ERROR(func_name);
    }
}
#undef THROW_FUNC_ERROR

}  // namespace lite
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
