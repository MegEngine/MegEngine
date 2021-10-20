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

using namespace mgb;
using namespace gopt;

namespace {
using OprFormat = LayoutTransformContext::OprFormat;
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
        OprFormat base_opr_format, TensorFormats base_tensor_format) {
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
            base_opr_format, base_tensor_format, Target::CUDA,
            LayoutTransformContext::ReformatAttribute::AUTO_PADDING_NHWC};
    auto ctx = std::make_unique<LayoutTransformContext>(
            std::move(opr_list), std::move(available_tensor_formats), attribute);
    ctx->add_opr_config(
               opr::ConvBiasForward::typeinfo(),
               {OprFormat::NCHW, OprFormat::NHWC, OprFormat::NCHW4, OprFormat::NCHW32,
                OprFormat::NCHW64, OprFormat::CHWN4})
            .add_opr_config(
                    opr::ConvolutionForward::typeinfo(),
                    {OprFormat::NCHW, OprFormat::NCHW4})
            .add_opr_config(
                    opr::ConvolutionBackwardData::typeinfo(),
                    {OprFormat::NCHW, OprFormat::NCHW4, OprFormat::NHWC})
            .add_opr_config(
                    opr::PoolingForward::typeinfo(),
                    {OprFormat::NCHW4, OprFormat::NCHW32, OprFormat::NHWC,
                     OprFormat::NCHW64, OprFormat::CHWN4})
            .add_opr_config(
                    opr::WarpPerspectiveForward::typeinfo(),
                    {OprFormat::NHWC, OprFormat::NCHW4, OprFormat::NCHW64});
    return ctx;
}
}  // namespace

/* ================= LayoutTransformContext ==================*/
LayoutTransformContext& LayoutTransformContext::add_opr_config(
        Typeinfo* opr, OprFormat opr_format) {
    auto& dispatchers = m_opr_configs[opr];
    dispatchers[opr_format] =
            OprTensorFormatsConfiguration::find_dispatcher_by_type_format(
                    opr, opr_format);
    return *this;
}

LayoutTransformContext& LayoutTransformContext::add_opr_config(
        Typeinfo* opr, SmallVector<OprFormat> opr_formats) {
    auto& dispatchers = m_opr_configs[opr];
    for (auto opr_fmt : opr_formats) {
        dispatchers[opr_fmt] =
                OprTensorFormatsConfiguration::find_dispatcher_by_type_format(
                        opr, opr_fmt);
    }
    return *this;
}

std::unique_ptr<LayoutTransformContext> LayoutTransformContext::make(
        Target target, OprFormat base_opr_format, TensorFormats base_tensor_format) {
    switch (target) {
        case Target::CUDA:
            return make_cuda_ctx(base_opr_format, base_tensor_format);
        default:
            mgb_assert(false, "unsupported target %s\n", target_to_string(target));
    }
}

// vim: syntax=cpp.doxygen
