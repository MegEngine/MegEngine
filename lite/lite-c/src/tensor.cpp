/**
 * \file lite-c/src/tensor.cpp
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#include "lite/tensor.h"
#include "../../src/tensor_impl_base.h"
#include "common.h"
#include "lite-c/tensor_c.h"
#include <set>
#include <string>
#include <unordered_map>

const LiteLayout default_layout = {.shapes = {0, 0, 0, 0, 0},
                                   .ndim = 0,
                                   .data_type = LiteDataType::LITE_FLOAT};

const LiteTensorDesc default_desc = {.is_pinned_host = false,
                                     .layout = default_layout,
                                     .device_type = LiteDeviceType::LITE_CPU,
                                     .device_id = 0};
namespace {
std::unordered_map<void*, std::shared_ptr<lite::Tensor>>&
get_global_tensor_holder() {
    static thread_local std::unordered_map<void*, std::shared_ptr<lite::Tensor>>
            global_holder;
    return global_holder;
}
std::unordered_map<std::string, lite::LiteAny>&
get_global_tensor_attr_holder() {
    static thread_local std::unordered_map<std::string, lite::LiteAny>
            global_holder;
    return global_holder;
}
}  // namespace

//! convert the lite::Layout to Layout
LiteLayout convert_to_clayout(const lite::Layout& layout) {
    LiteLayout clayout;
    clayout.ndim = layout.ndim;
    LITE_ASSERT(layout.ndim < LAYOUT_MAX_DIM, "layout ndim is to large");
    for (size_t i = 0; i < layout.ndim; i++) {
        clayout.shapes[i] = layout.shapes[i];
    }
    clayout.data_type = layout.data_type;
    return clayout;
}

//! convert the C Layout to lite::Layout
lite::Layout convert_to_layout(const LiteLayout& clayout) {
    lite::Layout layout;
    layout.ndim = clayout.ndim;
    LITE_ASSERT(layout.ndim < LAYOUT_MAX_DIM, "clayout ndim is to large");
    for (size_t i = 0; i < layout.ndim; i++) {
        layout.shapes[i] = clayout.shapes[i];
    }
    layout.data_type = clayout.data_type;
    return layout;
}

int LITE_make_tensor(const LiteTensorDesc tensor_describe, LiteTensor* tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE_make_tensor is null");
    lite::Layout layout = convert_to_layout(tensor_describe.layout);
    auto lite_tensor = std::make_shared<lite::Tensor>(
            tensor_describe.device_id, tensor_describe.device_type, layout,
            tensor_describe.is_pinned_host);
    get_global_tensor_holder()[lite_tensor.get()] = lite_tensor;
    *tensor = lite_tensor.get();
    LITE_CAPI_END();
}

int LITE_destroy_tensor(LiteTensor tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    get_global_tensor_holder().erase(tensor);
    LITE_CAPI_END();
}

int LITE_set_tensor_layout(LiteTensor tensor, const LiteLayout layout) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    auto tensor_ptr = static_cast<lite::Tensor*>(tensor);
    tensor_ptr->set_layout(convert_to_layout(layout));
    LITE_CAPI_END();
}

int LITE_reset_tensor_memory(LiteTensor tensor, void* prepared_data,
                             size_t data_length_in_byte) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(prepared_data, "The prepared_data pass to LITE c_api is null");
    static_cast<lite::Tensor*>(tensor)->reset(prepared_data,
                                              data_length_in_byte);
    LITE_CAPI_END();
}

int LITE_reset_tensor(LiteTensor tensor, const LiteLayout layout,
                      void* prepared_data) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(prepared_data, "The prepared_data pass to LITE c_api is null");
    static_cast<lite::Tensor*>(tensor)->reset(prepared_data,
                                              convert_to_layout(layout));
    LITE_CAPI_END();
}

int LITE_tensor_reshape(LiteTensor tensor, const int* shape, int size) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor && shape, "The tensor pass to LITE c_api is null");
    std::vector<int> shapes;
    for (int i = 0; i < size; i++) {
        shapes.push_back(shape[i]);
    }
    static_cast<lite::Tensor*>(tensor)->reshape(shapes);
    LITE_CAPI_END();
}

int LITE_tensor_slice(const LiteTensor tensor, const size_t* start,
                      const size_t* end, const size_t* step, size_t size,
                      LiteTensor* slice_tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor && start && end && slice_tensor,
                "The tensor pass to LITE c_api is null");
    std::vector<size_t> starts, ends, steps;
    for (size_t i = 0; i < size; i++) {
        starts.push_back(start[i]);
        ends.push_back(end[i]);
        if (step) {
            steps.push_back(step[i]);
        }
    }
    auto ret_tensor =
            static_cast<lite::Tensor*>(tensor)->slice(starts, ends, steps);
    get_global_tensor_holder()[ret_tensor.get()] = ret_tensor;
    *slice_tensor = ret_tensor.get();
    LITE_CAPI_END();
}

int LITE_tensor_fill_zero(LiteTensor tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    static_cast<lite::Tensor*>(tensor)->fill_zero();
    LITE_CAPI_END();
}

int LITE_tensor_copy(LiteTensor dst_tensor, const LiteTensor src_tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(dst_tensor && src_tensor,
                "The tensor pass to LITE c_api is null");
    static_cast<lite::Tensor*>(dst_tensor)
            ->copy_from(*static_cast<lite::Tensor*>(src_tensor));
    LITE_CAPI_END();
}

int LITE_tensor_share_memory_with(LiteTensor dst_tensor,
                                  const LiteTensor src_tensor) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(dst_tensor && src_tensor,
                "The tensor pass to LITE c_api is null");
    static_cast<lite::Tensor*>(dst_tensor)
            ->share_memory_with(*static_cast<lite::Tensor*>(src_tensor));
    LITE_CAPI_END();
}

int LITE_get_tensor_memory(const LiteTensor tensor, void** data) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(data, "The data ptr pass to LITE c_api is null");
    *data = static_cast<lite::Tensor*>(tensor)->get_memory_ptr();
    LITE_CAPI_END();
}

int LITE_get_tensor_memory_with_index(const LiteTensor tensor,
                                      const size_t* index, size_t size,
                                      void** data) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor && index && data,
                "The tensor pass to LITE c_api is null");
    std::vector<size_t> index_v;
    for (size_t i = 0; i < size; i++) {
        index_v.push_back(index[i]);
    }
    *data = static_cast<lite::Tensor*>(tensor)->get_memory_ptr(index_v);
    LITE_CAPI_END();
}

int LITE_get_tensor_total_size_in_byte(const LiteTensor tensor, size_t* size) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(size, "The size ptr pass to LITE c_api is null");
    *size = static_cast<lite::Tensor*>(tensor)->get_tensor_total_size_in_byte();
    LITE_CAPI_END();
}

int LITE_get_tensor_layout(const LiteTensor tensor, LiteLayout* layout) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(layout, "The layout ptr pass to LITE c_api is null");
    *layout = convert_to_clayout(
            static_cast<lite::Tensor*>(tensor)->get_layout());
    LITE_CAPI_END();
}

int LITE_get_tensor_device_type(const LiteTensor tensor,
                           LiteDeviceType* device_type) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(device_type, "The device ptr pass to LITE c_api is null");
    *device_type = static_cast<lite::Tensor*>(tensor)->get_device_type();
    LITE_CAPI_END();
}

int LITE_get_tensor_device_id(const LiteTensor tensor, int* device_id) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor && device_id, "The tensor pass to LITE c_api is null");
    *device_id = static_cast<lite::Tensor*>(tensor)->get_device_id();
    LITE_CAPI_END();
}

int LITE_is_pinned_host(const LiteTensor tensor, int* is_pinned_host) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(is_pinned_host,
                "The is_pinned_host ptr pass to LITE c_api is null");
    *is_pinned_host = static_cast<lite::Tensor*>(tensor)->is_pinned_host();
    LITE_CAPI_END();
}

int LITE_is_memory_continue(const LiteTensor tensor, int* is_continue) {
    LITE_CAPI_BEGIN();
    LITE_ASSERT(tensor, "The tensor pass to LITE c_api is null");
    LITE_ASSERT(is_continue, "The is_continue ptr pass to LITE c_api is null");
    *is_continue = static_cast<lite::Tensor*>(tensor)->is_continue_memory();
    LITE_CAPI_END();
}

int LITE_tensor_concat(LiteTensor* tensors, int nr_tensor, int dim,
                       LiteDeviceType dst_device, int device_id,
                       LiteTensor* result_tensor) {
    LITE_CAPI_BEGIN();
    std::vector<lite::Tensor> v_tensors;
    for (int i = 0; i < nr_tensor; i++) {
        v_tensors.push_back(*static_cast<lite::Tensor*>(tensors[i]));
    }
    auto tensor =
            lite::TensorUtils::concat(v_tensors, dim, dst_device, device_id);
    get_global_tensor_holder()[tensor.get()] = tensor;
    *result_tensor = tensor.get();
    LITE_CAPI_END()
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
