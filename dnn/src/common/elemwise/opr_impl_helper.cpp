/**
 * \file dnn/src/common/elemwise/opr_impl_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./opr_impl_helper.h"
#include "src/common/utils.h"

using namespace megdnn;

template <int arity>
ElemwiseOpParamN<arity> ElemwiseLayoutHelper::make_elemwise_op_param(
        void* opr,
        void (*check_layout_and_broadcast)(
                void*, const TensorLayoutPtrArray&, const TensorLayout&),
        const TensorNDArray& src, const TensorND& dst) {
    megdnn_assert(src.size() == static_cast<size_t>(arity));
    ElemwiseOpParamN<arity> ret;
    TensorLayoutPtrArray src_layouts(arity);
    for (int i = 0; i < arity; ++i) {
        ret.param[i] = src[i];
        src_layouts[i] = &ret.param[i].layout;
    }
    check_layout_and_broadcast(opr, src_layouts, dst.layout);
    ret.init_from_given_tensor();
    return ret;
}

// explicit instantiation so subclasses can call this method
#define INST(n)                                                                       \
    template ElemwiseOpParamN<n> ElemwiseLayoutHelper::make_elemwise_op_param<n>(     \
            void*, void (*)(void*, const TensorLayoutPtrArray&, const TensorLayout&), \
            const TensorNDArray&, const TensorND&)
INST(1);
INST(2);
INST(3);
INST(4);
INST(5);
INST(6);
#undef INST

void ElemwiseForwardImplHelper::prepare_fma3(
        ElemwiseOpParamN<3>& param, bool& c_is_scalar) {
    c_is_scalar = is_broadcasted_scalar(m_src->at(2).layout);
    param = make_elemwise_op_param<3>();

    if (!c_is_scalar && !param[2].layout.eq_layout(param[0].layout)) {
        megdnn_assert_eq_layout(param[2].layout, param[1].layout);
        std::swap(param[0], param[1]);
    }
    if (c_is_scalar && param[2].layout.eq_layout(param[0].layout)) {
        std::swap(param[0], param[1]);
    }
}

void ElemwiseForwardImplHelper::prepare_fma4(ElemwiseOpParamN<4>& param) {
    param = make_elemwise_op_param<4>();
    if (!param[0].layout.eq_layout(param[2].layout))
        std::swap(param[0], param[1]);

    megdnn_assert_eq_layout(param[0].layout, param[2].layout);
    megdnn_assert_eq_layout(param[1].layout, param[3].layout);
}

bool ElemwiseLayoutHelper::is_broadcasted_scalar(const TensorLayout& layout) {
    if (layout.format.type() != TensorFormat::Type::DEFAULT)
        return false;
    for (size_t i = 0; i < layout.ndim; ++i) {
        if (layout.shape[i] != 1 && layout.stride[i] != 0)
            return false;
    }
    return true;
}
template <size_t slice_size>
bool ElemwiseLayoutHelper::is_broadcastedx_channel_like(
        const TensorLayout& layout, BroadcastChannelInfo& info) {
    if (layout.format.type() == TensorFormat::Type::DEFAULT && layout.ndim == 3 &&
        layout.stride[0] == slice_size && layout.stride[1] == 0 &&
        layout.stride[2] == 1) {
        info.x = layout.shape[0];
        info.y = layout.shape[1];
        info.z = layout.shape[2];
        return true;
    } else if (
            layout.format.type() == TensorFormat::Type::DEFAULT && layout.ndim == 4 &&
            layout.stride[0] == 0 && layout.stride[1] == slice_size &&
            layout.stride[2] == 0 && layout.stride[3] == 1) {
        info.x = layout.shape[1];
        info.y = layout.shape[2];
        info.z = layout.shape[3];
        return true;
    }
    return false;
}
#define INST(n)                                                          \
    template bool ElemwiseLayoutHelper::is_broadcastedx_channel_like<n>( \
            const TensorLayout& layout, BroadcastChannelInfo& info)
INST(4);
INST(8);
#undef INST

bool ElemwiseLayoutHelper::is_broadcasted_channel_like(
        const TensorLayout& layout, BroadcastChannelInfo& info) {
    if (layout.format.type() == TensorFormat::Type::DEFAULT) {
        if (layout.ndim == 3 && layout.stride[0] == 0 && layout.stride[2] == 0 &&
            layout.stride[1] == 1) {
            info.x = layout.shape[0];
            info.y = layout.shape[1];
            info.z = layout.shape[2];
            return true;
        } else if (layout.ndim == 2 && layout.stride[1] == 0 && layout.stride[0] == 1) {
            info.x = 1;
            info.y = layout.shape[0];
            info.z = layout.shape[1];
            return true;
        }
    } else {
        if (Image2DPack4TensorFormat::is_valid_image(layout)) {
            auto align_axis =
                    layout.format.as_impl<Image2DPack4TensorFormat>().align_axis();
            if (layout.ndim == 4 && align_axis == 1 &&
                (layout.stride[0] == 0 || layout.shape[0] == 1) &&
                layout.stride[1] == 4 && layout.stride[2] == 0 &&
                layout.stride[3] == 1) {
                info.x = 1;
                info.y = 1;
                info.z = layout.shape[2];
                return true;
            } else if (
                    layout.ndim == 3 && align_axis == 1 &&
                    (layout.stride[0] == 0 || layout.shape[0] == 1) &&
                    layout.stride[1] == 0 && layout.shape[2] == 4 &&
                    layout.stride[2] == 1) {
                //! [1, 1, 1, 1, 4] + [N, H, 1, W, 4]
                info.x = 1;
                info.y = 1;
                info.z = layout.shape[1];
                return true;
            }
            return false;
        }
    }
    return false;
}

bool ElemwiseLayoutHelper::is_NHWC_broadcasted_channel_like(
        const TensorLayout& layout, BroadcastChannelInfo& info) {
    if (layout.format.type() == TensorFormat::Type::DEFAULT) {
        if (layout.ndim == 2 && layout.stride[1] == 1 && layout.stride[0] == 0) {
            info.x = 1;
            info.y = layout.shape[0];
            info.z = layout.shape[1];
            return true;
        }
    }
    return false;
}

bool ElemwiseLayoutHelper::is_broadcasted_1x(
        const TensorLayout& layout, Broadcast1xInfo& binfo) {
    if (layout.ndim == 2 && layout.stride[0] == 0 && layout.stride[1] == 1) {
        binfo.x = layout[0];
        binfo.y = layout[1];
        return true;
    }
    if (layout.ndim == 1 && layout.stride[0] == 1) {
        binfo.x = 1;
        binfo.y = layout[0];
        return true;
    }
    return false;
}

// vim: syntax=cpp.doxygen
