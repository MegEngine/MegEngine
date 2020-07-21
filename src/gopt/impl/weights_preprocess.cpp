/**
 * \file src/gopt/impl/weights_preprocess.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/gopt/weights_preprocess.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/tensor_manip.h"

#include "megbrain/utils/hash_ct.h"
#include "midout.h"

MIDOUT_DECL(megbrain_weight_preprocess)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_weight_preprocess, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace gopt;
using namespace cg;

const char* WinogradTransformReplacePass::name() const {
    return "winograd_transform";
}

void WinogradTransformReplacePass::apply(OptState& opt) const {
    MIDOUT_B("WinogradTransformReplacePass::apply")
    auto rewriter = opt.graph().make_rewriter();
    ConstVarPropogate cvprop{ConstVarType::IMMUTABLE_AND_PARAM};
    opt.graph().iter([&cvprop](OperatorNodeBase *opr) {
        cvprop.add_opr(opr);
    });

    auto get_algo = [](const opr::ConvBias& opr) -> std::string {
        auto&& inputs = opr.input();
        SmallVector<TensorLayout> layouts;
        mgb_assert(inputs.size() >= 2 && inputs.size() <= 4);
        auto&& mo = opr.megdnn_opr();
        for (size_t i = 0; i < 4; i++) {
            if (inputs.size() <= i) {
                if (i == 2) {
                    //! bias
                    DType dtype;
                    mo->deduce_dtype(inputs[0]->dtype(), inputs[1]->dtype(),
                                     DType{}, DType{}, dtype);
                    layouts.emplace_back(TensorShape{}, dtype);
                } else {
                    layouts.emplace_back(TensorShape{}, opr.output(0)->dtype(),
                                         opr.output(0)->format());
                }
            } else {
                layouts.emplace_back(inputs[i]->shape(), inputs[i]->dtype(),
                                     inputs[i]->format());
            }
        }
        layouts.emplace_back(opr.output(0)->shape(), opr.output(0)->dtype(),
                             opr.output(0)->format());

        AlgoChooserProfileCache& cache = opr.profile_cache();
        auto param_blob = opr.param_blob();
        AlgoChooserProfileCache::Key cache_key{layouts.data(), layouts.size(),
                                               param_blob.first,
                                               param_blob.second};
        auto&& rst = cache.get(cache_key);
        if (!rst.valid())
            return "";
        auto prof = rst.val();
        if (prof.empty())
            return "";
        return prof[0].algo;
    };
    auto on_opr = [&](OperatorNodeBase* opr) {
        auto type = opr->dyn_typeinfo();
        do {
            if (type != opr::ConvBias::typeinfo())
                break;
            auto&& conv_bias_opr = opr->cast_final_safe<opr::ConvBias>();
            auto&& inputs = conv_bias_opr.input();
            VarNodeArray new_inp;
            new_inp.reserve(inputs.size());
            for (auto i : inputs) {
                new_inp.push_back(rewriter.get_var(i));
            }
            if (!(cvprop.is_midconst(inputs[1]) ||
                  cvprop.is_const(inputs[1]))) {
                break;
            }
            auto algo_name = get_algo(conv_bias_opr);
            auto winograd_param =
                    megdnn::ConvBias::parse_winograd_name(algo_name);
            if (winograd_param == megdnn::ConvBias::INVALID_WINOGRAD_PARAM)
                break;
            mgb_assert(
                    conv_bias_opr.param().format ==
                                    megdnn::ConvBias::Param::Format::NCHW ||
                            conv_bias_opr.param().format ==
                                    megdnn::ConvBias::Param::Format::NCHW88 ||
                            conv_bias_opr.param().format ==
                                    megdnn::ConvBias::Param::Format::NCHW44,
                    "currently winograd only suppport NCHW and NCHW44 and "
                    "NCHW88");
            opr::ConvBiasForward::check_winograd_param_valid(
                    winograd_param, conv_bias_opr.input(0)->dtype());
            megdnn::param::Winograd winograd_preprocess_param;
            winograd_preprocess_param.format =
                    opr::ConvBiasForward::get_matmul_format(winograd_param);
            winograd_preprocess_param.output_block_size =
                    winograd_param.output_block_size;

            auto conv_bias_param = conv_bias_opr.param();
            //! If input dtype is Qint8 and matmul format is MK4, The winograd
            //! compute type is float.
            if (conv_bias_opr.input(0)->dtype().enumv() ==
                        DTypeEnum::QuantizedS8 &&
                winograd_preprocess_param.format ==
                        megdnn::param::MatrixMul::Format::MK4) {
                winograd_preprocess_param.compute_mode =
                        megdnn::param::ConvBias::ComputeMode::FLOAT32;
                conv_bias_param.compute_mode =
                        megdnn::param::ConvBias::ComputeMode::FLOAT32;
            }

            auto winograd_preprocess_opr = opr::WinogradFilterPreprocess::make(
                    new_inp[1], winograd_preprocess_param);
            mgb_assert(inputs.size() == 2 || inputs.size() == 3,
                       "input size need to be 2/3, but got: %zu",
                       inputs.size());
            SymbolVar new_conv_bias_opr;

            if (new_inp[0]->shape().ndim == 4) {
                conv_bias_param.format =
                        megdnn::ConvBias::Param::Format::NCHW_WINOGRAD;
            } else {
                mgb_assert(new_inp[0]->shape().ndim == 5);
                size_t pack_size = new_inp[0]->shape()[4];
                if (pack_size == 8) {
                    conv_bias_param.format =
                            megdnn::ConvBias::Param::Format::NCHW88_WINOGRAD;
                } else if (pack_size == 4) {
                    conv_bias_param.format =
                            megdnn::ConvBias::Param::Format::NCHW44_WINOGRAD;
                } else {
                    mgb_assert(0, "Invalid pack size %zu in algo %s", pack_size,
                               algo_name.c_str());
                }
            }

            conv_bias_param.output_block_size =
                    winograd_param.output_block_size;
            if (inputs.size() == 2) {
                new_conv_bias_opr = opr::ConvBias::make(
                        new_inp[0], winograd_preprocess_opr.node(),
                        conv_bias_param, conv_bias_opr.execution_policy(),
                        conv_bias_opr.config());
            } else {
                new_conv_bias_opr = opr::ConvBias::make(
                        new_inp[0], winograd_preprocess_opr.node(), new_inp[2],
                        conv_bias_param, conv_bias_opr.execution_policy(),
                        conv_bias_opr.config());
            }

            auto&& origin_out = conv_bias_opr.output();
            auto&& cur_out = new_conv_bias_opr.node()->owner_opr()->output();
            mgb_assert(origin_out.size() == cur_out.size());
            for (size_t i = 0; i < origin_out.size(); i++) {
                if (!origin_out[i]->contain_flag(
                            VarNode::Flag::VOLATILE_CONTENT)) {
                    rewriter.replace_var(origin_out[i], cur_out[i], nullptr);
                }
            }
            return;
        } while (0);

        rewriter.auto_replace_outputs(opr);
    };

    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
    MIDOUT_E
}

/**
 * \warning WinogradTransformReplacePass implies that we run ParamFuse pass
 * before(currently run ParamFuse in optimize_for_inference when dump model),
 * othwise it can not deal with \c ConvBias(x, W+1), as the node of W+1 has no
 * flag PERSISTENT_DEVICE_VALUE, it's a mid-const node, we should use
 * ConstVarPropogate strictly speaking.
 */
void gopt::transform_vars_inplace_with_winograd(
        mgb::cg::VarNodeArray& dest_vars) {
    gopt::GraphOptimizer optimizer;
    optimizer.add_pass<WinogradTransformReplacePass>();
    optimizer.add_pass<ParamFusePass>();
    optimizer.apply_inplace(dest_vars);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
