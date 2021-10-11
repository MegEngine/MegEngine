/**
 * \file src/mge/common.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "lite_build_config.h"

#if LITE_BUILD_WITH_MGE
#include "common.h"
#include "megdnn/dtype.h"

using namespace lite;
using namespace mgb;

enum class CompressionMethod {
    NO_COMPRESSION = 0,
    FLOAT32_STRIDE_FLOAT32_BASE_UINT8_WEIGHTS = 1,
    FLOAT32_STRIDE_FLOAT32_BASE_UINT16_WEIGHTS = 2,
};

void lite::decompressed_tensor_value_loader(
        void* ptr_, const mgb::TensorLayout& layout,
        mgb::serialization::InputFile& fin) {
    uint8_t compress_flag;
    fin.read(&compress_flag, sizeof(compress_flag));
    size_t num_weights = layout.total_nr_elems();
    switch (CompressionMethod(compress_flag)) {
        case CompressionMethod::NO_COMPRESSION: {
            mgb::serialization::GraphLoadConfig::default_tensor_value_loader(
                    ptr_, layout, fin);
            break;
        }
        case CompressionMethod::FLOAT32_STRIDE_FLOAT32_BASE_UINT8_WEIGHTS: {
            if (ptr_) {
                float stride, base;
                std::vector<uint8_t> weights(num_weights);
                fin.read(&stride, sizeof(stride));
                fin.read(&base, sizeof(base));
                fin.read(weights.data(), num_weights * sizeof(uint8_t));
                auto* ptr = static_cast<float*>(ptr_);
                for (size_t i = 0; i < num_weights; ++i)
                    ptr[i] = stride * weights[i] + base;
            } else {
                fin.skip(sizeof(float) * 2 + num_weights * sizeof(uint8_t));
            }
            break;
        }
        case CompressionMethod::FLOAT32_STRIDE_FLOAT32_BASE_UINT16_WEIGHTS: {
            if (ptr_) {
                float stride, base;
                std::vector<uint16_t> weights(num_weights);
                fin.read(&stride, sizeof(stride));
                fin.read(&base, sizeof(base));
                fin.read(weights.data(), num_weights * sizeof(uint16_t));
                auto* ptr = static_cast<float*>(ptr_);
                for (size_t i = 0; i < num_weights; ++i)
                    ptr[i] = stride * weights[i] + base;
            } else {
                fin.skip(sizeof(float) * 2 + num_weights * sizeof(uint16_t));
            }
            break;
        }
        default:
            LITE_THROW("Unexpected compression method");
    }
}

LTensorLayout lite::to_impl_layout(const Layout& layout) {
    mgb::TensorLayout mge_layout;
    mge_layout.ndim = layout.ndim;
    LITE_ASSERT(layout.ndim < TensorShape::MAX_NDIM, "lite layout ndim is to large");
    for (size_t i = 0; i < layout.ndim; i++) {
        mge_layout.shape[i] = layout.shapes[i];
    }
    mge_layout.init_contiguous_stride();
    switch (layout.data_type) {
        case LiteDataType::LITE_FLOAT:
            mge_layout.dtype = mgb::dtype::Float32();
            break;
#if !MEGDNN_DISABLE_FLOAT16
        case LiteDataType::LITE_HALF:
            mge_layout.dtype = mgb::dtype::Float16();
            break;
#endif
        case LiteDataType::LITE_INT:
            mge_layout.dtype = mgb::dtype::Int32();
            break;
        case LiteDataType::LITE_INT8:
            mge_layout.dtype = mgb::dtype::Int8();
            break;
        case LiteDataType::LITE_UINT8:
            mge_layout.dtype = mgb::dtype::Uint8();
            break;
        case LiteDataType::LITE_INT16:
            mge_layout.dtype = mgb::dtype::Int16();
            break;
        case LiteDataType::LITE_UINT16:
            mge_layout.dtype = mgb::dtype::Uint16();
            break;
        default:
            LITE_THROW(mgb::ssprintf(
                    "unsupport dtype in lite enum id is %d.",
                    static_cast<int>(layout.data_type)));
    }
    return mge_layout;
}

Layout lite::to_lite_layout(const LTensorLayout& mge_layout) {
    Layout layout;
    if (!mge_layout.dtype.valid()) {
        return layout;
    }
    layout.ndim = mge_layout.ndim;
    LITE_ASSERT(layout.ndim < layout.MAXDIM, "tensor layout ndim is to large");
    for (size_t i = 0; i < layout.ndim; i++) {
        layout.shapes[i] = mge_layout.shape[i];
    }
    switch (mge_layout.dtype.enumv()) {
        case mgb::DTypeEnum::Float32:
            layout.data_type = LiteDataType::LITE_FLOAT;
            break;
#if !MEGDNN_DISABLE_FLOAT16
        case mgb::DTypeEnum::Float16:
            layout.data_type = LiteDataType::LITE_HALF;
            break;
#endif
        case mgb::DTypeEnum::Int32:
            layout.data_type = LiteDataType::LITE_INT;
            break;
        case mgb::DTypeEnum::Int16:
            layout.data_type = LiteDataType::LITE_INT16;
            break;
        case mgb::DTypeEnum::Uint16:
            layout.data_type = LiteDataType::LITE_UINT16;
            break;
        case mgb::DTypeEnum::Int8:
            layout.data_type = LiteDataType::LITE_INT8;
            break;
        case mgb::DTypeEnum::Uint8:
            layout.data_type = LiteDataType::LITE_UINT8;
            break;
        default:
            LITE_THROW(mgb::ssprintf(
                    "unsupport dtype in lite : %s.", mge_layout.to_string().c_str()));
    }
    return layout;
}

mgb::CompNode::Locator lite::to_compnode_locator(const LiteDeviceType& device) {
    mgb::CompNode::Locator loc;
    switch (device) {
        case LiteDeviceType::LITE_CPU:
            loc.type = mgb::CompNode::DeviceType::CPU;
            break;
        case LiteDeviceType::LITE_CUDA:
            loc.type = mgb::CompNode::DeviceType::CUDA;
            break;
        case LiteDeviceType::LITE_ATLAS:
            loc.type = mgb::CompNode::DeviceType::ATLAS;
            break;
        case LiteDeviceType::LITE_DEVICE_DEFAULT:
            loc.type = mgb::CompNode::DeviceType::UNSPEC;
            break;
        default:
            LITE_THROW(ssprintf(
                    "lite unsupported compnode type: enum value: %d.", (int)(device)));
    }
    return loc;
}

LiteDeviceType lite::get_device_from_locator(const mgb::CompNode::Locator& locator) {
    switch (locator.type) {
        case mgb::CompNode::DeviceType::CPU:
        case mgb::CompNode::DeviceType::MULTITHREAD:
            return LiteDeviceType::LITE_CPU;
        case mgb::CompNode::DeviceType::CUDA:
            return LiteDeviceType::LITE_CUDA;
        case mgb::CompNode::DeviceType::ATLAS:
            return LiteDeviceType::LITE_ATLAS;
        case mgb::CompNode::DeviceType::UNSPEC:
            return LiteDeviceType::LITE_DEVICE_DEFAULT;
        default:
            LITE_THROW(ssprintf(
                    "lite unsupported compnode type: enum value: %d.",
                    (int)(locator.type)));
    }
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
