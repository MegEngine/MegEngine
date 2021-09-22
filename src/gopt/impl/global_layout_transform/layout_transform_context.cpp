/**
 * \file src/gopt/impl/layout_transform_context.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/gopt/layout_transform_context.h"
#include "./utils.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_manip.h"

using namespace mgb;
using namespace gopt;

namespace {
using OprFormat = LayoutTransformContext::OprFormat;
using OprFormatConfigID = LayoutTransformContext::OprFormatConfigID;
using OprList = LayoutTransformContext::OprList;
using Attribute = LayoutTransformContext::Attribute;
using Target = LayoutTransformContext::Target;
const char* target_to_string(Target target) {
#define cb(_target)       \
    case Target::_target: \
        return #_target
    switch (target) {
        cb(CUDA);
        cb(X86);
        cb(ARM);
        cb(UNSPEC);
        default:
            mgb_assert(
                    false, "unsupported target (got:%u)",
                    static_cast<uint32_t>(target));
    }
#undef cb
}

std::unique_ptr<LayoutTransformContext> make_cuda_ctx(
        OprFormatConfigID base_config_id, TensorFormats base_tensor_format) {
    OprList opr_list = {
            opr::ConvBiasForward::typeinfo(),
            opr::ConvolutionForward::typeinfo(),
            opr::ConvolutionBackwardData::typeinfo(),
            opr::ElemwiseMultiType::typeinfo(),
            opr::Elemwise::typeinfo(),
            opr::TypeCvt::typeinfo(),
            opr::PoolingForward::typeinfo(),
            opr::WarpPerspectiveForward::typeinfo(),
    };

    SmallVector<TensorFormats> available_tensor_formats = {
            TensorFormats::NCHW,    TensorFormats::NHWC,    TensorFormats::NCHWc4,
            TensorFormats::NCHWc32, TensorFormats::NCHWc64, TensorFormats::CHWNc4};
    Attribute attribute = {
            base_config_id, base_tensor_format, Target::CUDA,
            LayoutTransformContext::ReformatAttribute::AUTO_PADDING_NHWC};
    auto ctx = std::make_unique<LayoutTransformContext>(
            std::move(opr_list), std::move(available_tensor_formats), attribute);
    ctx->add_opr_config(
               opr::ConvBiasForward::typeinfo(),
               {OprFormatConfigID::NCHW, OprFormatConfigID::NHWC,
                OprFormatConfigID::NCHW4_NCHW32, OprFormatConfigID::NCHW32_NCHW4,
                OprFormatConfigID::NCHW4, OprFormatConfigID::NCHW32,
                OprFormatConfigID::NCHW64, OprFormatConfigID::CHWN4})
            .add_opr_config(
                    opr::ConvolutionForward::typeinfo(),
                    {OprFormatConfigID::NCHW, OprFormatConfigID::NCHW4})
            .add_opr_config(
                    opr::ConvolutionBackwardData::typeinfo(),
                    {OprFormatConfigID::NCHW, OprFormatConfigID::NCHW4,
                     OprFormatConfigID::NHWC})
            .add_opr_config(
                    opr::PoolingForward::typeinfo(),
                    {OprFormatConfigID::NCHW4, OprFormatConfigID::NCHW32,
                     OprFormatConfigID::NHWC, OprFormatConfigID::NCHW64,
                     OprFormatConfigID::CHWN4})
            .add_opr_config(
                    opr::WarpPerspectiveForward::typeinfo(),
                    {OprFormatConfigID::NHWC, OprFormatConfigID::NCHW4,
                     OprFormatConfigID::NCHW64});
    return ctx;
}

std::unique_ptr<LayoutTransformContext> make_arm_ctx(
        OprFormatConfigID base_config_id, TensorFormats base_tensor_format) {
    OprList opr_list = {
            opr::ConvBiasForward::typeinfo(),
            opr::ConvolutionForward::typeinfo(),
            opr::ElemwiseMultiType::typeinfo(),
            opr::Elemwise::typeinfo(),
            opr::TypeCvt::typeinfo(),
            opr::PoolingForward::typeinfo(),
            opr::Resize::typeinfo(),
            opr::PowC::typeinfo(),
            opr::Concat::typeinfo(),
    };

    SmallVector<TensorFormats> available_tensor_formats = {
            TensorFormats::NCHW, TensorFormats::NCHWc4,
            DNN_INC_FLOAT16(TensorFormats::NCHWc8)};
    Attribute attribute = {base_config_id, base_tensor_format, Target::ARM};
    auto ctx = std::make_unique<LayoutTransformContext>(
            std::move(opr_list), std::move(available_tensor_formats), attribute);
    ctx->add_opr_config(
               opr::ConvBiasForward::typeinfo(),
               {OprFormatConfigID::NCHW, OprFormatConfigID::NCHW44,
                OprFormatConfigID::NCHW44_HYBRID,
                DNN_INC_FLOAT16(OprFormatConfigID::NCHW88),
                DNN_INC_FLOAT16(OprFormatConfigID::NCHW88_HYBRID),
                OprFormatConfigID::NCHW44_DOT, OprFormatConfigID::NCHW44_DOT_HYBRID})
            .add_opr_config(
                    opr::ConvolutionForward::typeinfo(),
                    {OprFormatConfigID::NCHW, OprFormatConfigID::NCHW44,
                     OprFormatConfigID::NCHW44_HYBRID,
                     DNN_INC_FLOAT16(OprFormatConfigID::NCHW88),
                     DNN_INC_FLOAT16(OprFormatConfigID::NCHW88_HYBRID),
                     OprFormatConfigID::NCHW44_DOT,
                     OprFormatConfigID::NCHW44_DOT_HYBRID})
            .add_opr_config(
                    opr::PoolingForward::typeinfo(),
                    {OprFormatConfigID::NCHW, OprFormatConfigID::NCHW44,
                     DNN_INC_FLOAT16(OprFormatConfigID::NCHW88)})
            .add_opr_config(
                    opr::ResizeForward::typeinfo(),
                    {OprFormatConfigID::NCHW, OprFormatConfigID::NCHW44,
                     DNN_INC_FLOAT16(OprFormatConfigID::NCHW88)});
    return ctx;
}
}  // namespace

/* ================= LayoutTransformContext ==================*/
LayoutTransformContext& LayoutTransformContext::add_opr_config(
        Typeinfo* opr, OprFormatConfigID config_id) {
    auto& dispatchers = m_opr_configs[opr];
    dispatchers[config_id] =
            OprTensorFormatsConfiguration::find_dispatcher_by_type_format(
                    opr, config_id);
    return *this;
}

LayoutTransformContext& LayoutTransformContext::add_opr_config(
        Typeinfo* opr, SmallVector<OprFormatConfigID> config_ids) {
    auto& dispatchers = m_opr_configs[opr];
    for (auto cfg : config_ids) {
        dispatchers[cfg] =
                OprTensorFormatsConfiguration::find_dispatcher_by_type_format(opr, cfg);
    }
    return *this;
}

std::unique_ptr<LayoutTransformContext> LayoutTransformContext::make(
        Target target, OprFormatConfigID base_config_id,
        TensorFormats base_tensor_format) {
    switch (target) {
        case Target::CUDA:
            return make_cuda_ctx(base_config_id, base_tensor_format);
        case Target::ARM:
            return make_arm_ctx(base_config_id, base_tensor_format);
        default:
            mgb_assert(false, "unsupported target %s\n", target_to_string(target));
    }
}

// vim: syntax=cpp.doxygen
