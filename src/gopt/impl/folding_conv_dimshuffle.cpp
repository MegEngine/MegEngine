/**
 * \file src/gopt/impl/folding_conv_dimshuffle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/gopt/inference.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/opr_shallow_copy.h"

#include "megdnn/opr_param_defs.h"

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "megbrain/utils/hash_ct.h"

#include "midout.h"

#include "megbrain/gopt/reformat_manager.h"

#if CUDA_VERSION >= 10020
MIDOUT_DECL(megbrain_folding_conv_dimshuffle)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_folding_conv_dimshuffle, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace gopt;
using ReformatKey = ReformatManager::ReformatKey;

/* ==================== FoldingConvBiasDimshufflePass ================= */
const char* FoldingConvBiasDimshufflePass::name() const {
    return mgb_cstr_log("folding conv bias dimshuffle pass");
}

void FoldingConvBiasDimshufflePass::apply(OptState& opt) const {
    MIDOUT_B("FoldingConvBiasDimshufflePass::apply");
    using DepType = cg::OperatorNodeProp::DepType;
    ThinHashMap<OperatorNodeBase*, SmallVector<std::pair<OperatorNodeBase*, DepType>>>
            readers;
    static const ThinHashSet<Typeinfo*> opr_type_list = {
            opr::TypeCvt::typeinfo(), opr::Dimshuffle::typeinfo(),
            opr::Reshape::typeinfo(), opr::ConvBias::typeinfo()};
    opt.graph().iter([&readers](OperatorNodeBase* opr) {
        for (auto&& i : opr->node_prop().dep_map()) {
            if (opr_type_list.count(i.first->owner_opr()->dyn_typeinfo())) {
                readers[i.first->owner_opr()].emplace_back(opr, i.second);
            }
        }
    });

    auto rewriter = opt.graph().make_rewriter();
    auto try_conv_dimshuffle_reshape_typecvt = [&rewriter,
                                                &readers](OperatorNodeBase* opr) {
        ThinHashSet<OperatorNodeBase*> opr_set;
        ThinHashSet<OperatorNodeBase*> reader_set;
        // check typecvt
        auto typecvt = try_cast_as_op<opr::TypeCvt>(opr);
        if (typecvt == nullptr)
            return false;
        auto inp_dtype = typecvt->input(0)->dtype(),
             out_dtype = typecvt->output(0)->dtype();
        bool is_s82f32 = inp_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                         out_dtype.enumv() == DTypeEnum::Float32;
        if (!is_s82f32)
            return false;
        opr_set.insert(opr);

        // check reshape
        auto reshape = try_cast_as_op<opr::Reshape>(typecvt->input(0)->owner_opr());
        if (reshape == nullptr)
            return false;
        opr_set.insert(reshape);
        for (auto&& i : readers[reshape]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }

        // check shuffle
        auto shuffle = try_cast_as_op<opr::Dimshuffle>(reshape->input(0)->owner_opr());
        if (shuffle == nullptr)
            return false;
        auto&& param = shuffle->param();
        if (param.pattern_len != 5)
            return false;
        bool is_nchw42nchw = param.pattern[0] == 0 && param.pattern[1] == 1 &&
                             param.pattern[2] == 4 && param.pattern[3] == 2 &&
                             param.pattern[4] == 3 &&
                             shuffle->input(0)->shape()[4] == 4;
        if (!is_nchw42nchw)
            return false;
        opr_set.insert(shuffle);
        for (auto&& i : readers[shuffle]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }

        // check conv bias
        auto conv_bias = try_cast_as_op<opr::ConvBias>(shuffle->input(0)->owner_opr());
        if (conv_bias == nullptr)
            return false;
        inp_dtype = conv_bias->input(0)->dtype();
        bool is_s8nchw4 =
                inp_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                conv_bias->param().format == megdnn::param::ConvBias::Format::NCHW4;
        if (!is_s8nchw4)
            return false;
        if (conv_bias->input().size() != 3)
            return false;
        opr_set.insert(conv_bias);
        for (auto&& i : readers[conv_bias]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        for (auto reader : reader_set) {
            if (opr_set.count(reader) <= 0) {
                return false;
            }
        }
        auto src = rewriter.get_var(conv_bias->input(0)),
             filter = rewriter.get_var(conv_bias->input(1)),
             bias = rewriter.get_var(conv_bias->input(2));
        auto new_bias = ReformatManager::instance().get(
                ReformatKey{TensorFormats::NCHWc4, TensorFormats::NCHW})({bias});
        new_bias = opr::TypeCvt::make(new_bias, dtype::Float32()).node();
        auto new_param = conv_bias->param();
        new_param.format = megdnn::param::ConvBias::Format::NCHW4_NCHW;
        auto conv_bias_shuffle = opr::ConvBias::make(
                src, filter, new_bias, new_param, conv_bias->execution_policy(),
                OperatorNodeConfig{dtype::Float32()});
        rewriter.replace_var(
                opr->output(0), conv_bias_shuffle.node(),
                mgb_cstr_log("replace conv_bias + typecvt + "
                             "dimshuffle + "
                             "reshape to conv_bias(NCHW4_NCHW)"));
        return true;
    };

    auto try_conv_reformat_nchw42nchw32 = [&rewriter, &readers](OperatorNodeBase* opr) {
        ThinHashSet<OperatorNodeBase*> opr_set;
        ThinHashSet<OperatorNodeBase*> reader_set;
        // check reshape
        auto reshape1 = try_cast_as_op<opr::Reshape>(opr);
        if (reshape1 == nullptr)
            return false;
        opr_set.insert(opr);
        // check dimshuffle
        auto shuffle = try_cast_as_op<opr::Dimshuffle>(reshape1->input(0)->owner_opr());
        if (shuffle == nullptr)
            return false;
        auto&& param = shuffle->param();
        if (param.pattern_len != 6)
            return false;
        bool is_nchw42nchw32 = param.pattern[0] == 0 && param.pattern[1] == 1 &&
                               param.pattern[2] == 3 && param.pattern[3] == 4 &&
                               param.pattern[4] == 2 && param.pattern[5] == 5 &&
                               shuffle->output(0)->shape()[5] == 4 &&
                               shuffle->output(0)->shape()[4] == 8;
        if (!is_nchw42nchw32)
            return false;
        opr_set.insert(shuffle);
        for (auto&& i : readers[shuffle]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        // check reshape
        auto reshape2 = try_cast_as_op<opr::Reshape>(shuffle->input(0)->owner_opr());
        if (reshape2 == nullptr)
            return false;
        opr_set.insert(reshape2);
        for (auto&& i : readers[reshape2]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        // check conv bias
        auto conv_bias = try_cast_as_op<opr::ConvBias>(reshape2->input(0)->owner_opr());
        if (conv_bias == nullptr)
            return false;
        auto inp_dtype = conv_bias->input(0)->dtype();
        bool is_s8nchw4 =
                inp_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                conv_bias->param().format == megdnn::param::ConvBias::Format::NCHW4;
        if (!is_s8nchw4)
            return false;
        if (conv_bias->input().size() != 3)
            return false;
        opr_set.insert(conv_bias);
        for (auto&& i : readers[conv_bias]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        for (auto reader : reader_set) {
            if (opr_set.count(reader) <= 0) {
                return false;
            }
        }
        auto src = rewriter.get_var(conv_bias->input(0)),
             filter = rewriter.get_var(conv_bias->input(1)),
             bias = rewriter.get_var(conv_bias->input(2));
        auto new_bias = ReformatManager::instance().get(
                ReformatKey{TensorFormats::NCHWc4, TensorFormats::NCHWc32})({bias});
        auto new_param = conv_bias->param();
        new_param.format = megdnn::param::ConvBias::Format::NCHW4_NCHW32;
        auto conv_bias_shuffle = opr::ConvBias::make(
                src, filter, new_bias, new_param, conv_bias->execution_policy(),
                conv_bias->config());
        rewriter.replace_var(
                opr->output(0), conv_bias_shuffle.node(),
                mgb_cstr_log("replace conv_bias + "
                             "reformat to conv_bias(NCHW4_NCHW32)"));
        return true;
    };

    auto try_conv_reformat_nchw42nhwc = [&rewriter, &readers](OperatorNodeBase* opr) {
        ThinHashSet<OperatorNodeBase*> opr_set;
        ThinHashSet<OperatorNodeBase*> reader_set;
        // check typecvt
        auto typecvt = try_cast_as_op<opr::TypeCvt>(opr);
        if (typecvt == nullptr)
            return false;
        auto in_dtype = typecvt->input(0)->dtype(),
             out_dtype = typecvt->output(0)->dtype();
        bool is_s82s4 = in_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                        (out_dtype.enumv() == DTypeEnum::QuantizedS4 ||
                         out_dtype.enumv() == DTypeEnum::Quantized4Asymm);
        if (!is_s82s4)
            return false;
        opr_set.insert(typecvt);

        // check reshape
        auto reshape = try_cast_as_op<opr::Reshape>(typecvt->input(0)->owner_opr());
        if (reshape == nullptr)
            return false;
        opr_set.insert(reshape);
        for (auto&& i : readers[reshape]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }

        // check dimshuffle
        auto shuffle = try_cast_as_op<opr::Dimshuffle>(reshape->input(0)->owner_opr());
        if (shuffle == nullptr)
            return false;
        auto&& param = shuffle->param();
        if (param.pattern_len != 5)
            return false;
        bool is_nchw42nhwc = param.pattern[0] == 0 && param.pattern[1] == 2 &&
                             param.pattern[2] == 3 && param.pattern[3] == 1 &&
                             param.pattern[4] == 4 &&
                             shuffle->output(0)->shape()[4] == 4;
        if (!is_nchw42nhwc)
            return false;
        opr_set.insert(shuffle);
        for (auto&& i : readers[shuffle]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }

        // check conv bias
        auto conv_bias = try_cast_as_op<opr::ConvBias>(shuffle->input(0)->owner_opr());
        if (conv_bias == nullptr)
            return false;
        auto inp_dtype = conv_bias->input(0)->dtype();
        bool is_s8nchw4 =
                inp_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                conv_bias->param().format == megdnn::param::ConvBias::Format::NCHW4;
        if (!is_s8nchw4)
            return false;
        if (conv_bias->input().size() != 3)
            return false;
        opr_set.insert(conv_bias);
        for (auto&& i : readers[conv_bias]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        for (auto reader : reader_set) {
            if (opr_set.count(reader) <= 0) {
                return false;
            }
        }
        auto src = rewriter.get_var(conv_bias->input(0)),
             filter = rewriter.get_var(conv_bias->input(1)),
             bias = rewriter.get_var(conv_bias->input(2));
        auto new_bias = ReformatManager::instance().get(
                ReformatKey{TensorFormats::NCHWc4, TensorFormats::NHWC})({bias});
        auto new_param = conv_bias->param();
        new_param.format = megdnn::param::ConvBias::Format::NCHW4_NHWC;
        auto conv_bias_shuffle = opr::ConvBias::make(
                src, filter, new_bias, new_param, conv_bias->execution_policy(),
                OperatorNodeConfig{out_dtype});
        rewriter.replace_var(
                opr->output(0), conv_bias_shuffle.node(),
                mgb_cstr_log("replace conv_bias + "
                             "reformat to conv_bias(NCHW4_NHWC)"));
        return true;
    };

    auto try_conv_reformat_nchw322nchw4 = [&rewriter, &readers](OperatorNodeBase* opr) {
        ThinHashSet<OperatorNodeBase*> opr_set;
        ThinHashSet<OperatorNodeBase*> reader_set;
        // check reshape
        auto reshape1 = try_cast_as_op<opr::Reshape>(opr);
        if (reshape1 == nullptr)
            return false;
        opr_set.insert(opr);
        // check dimshuffle
        auto shuffle = try_cast_as_op<opr::Dimshuffle>(reshape1->input(0)->owner_opr());
        if (shuffle == nullptr)
            return false;
        auto&& param = shuffle->param();
        if (param.pattern_len != 6)
            return false;
        bool is_nchw322nchw4 = param.pattern[0] == 0 && param.pattern[1] == 1 &&
                               param.pattern[2] == 4 && param.pattern[3] == 2 &&
                               param.pattern[4] == 3 && param.pattern[5] == 5 &&
                               shuffle->input(0)->shape()[5] == 4 &&
                               shuffle->input(0)->shape()[4] == 8;
        if (!is_nchw322nchw4)
            return false;
        opr_set.insert(shuffle);
        for (auto&& i : readers[shuffle]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        // check reshape
        auto reshape2 = try_cast_as_op<opr::Reshape>(shuffle->input(0)->owner_opr());
        if (reshape2 == nullptr)
            return false;
        opr_set.insert(reshape2);
        for (auto&& i : readers[reshape2]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        // check conv bias
        auto conv_bias = try_cast_as_op<opr::ConvBias>(reshape2->input(0)->owner_opr());
        if (conv_bias == nullptr)
            return false;
        auto inp_dtype = conv_bias->input(0)->dtype();
        bool is_s8nchw32 =
                inp_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                conv_bias->param().format == megdnn::param::ConvBias::Format::NCHW32;
        if (!is_s8nchw32)
            return false;
        if (conv_bias->input().size() != 3)
            return false;
        opr_set.insert(conv_bias);
        for (auto&& i : readers[conv_bias]) {
            if (i.second & DepType::DEV_VALUE) {
                reader_set.insert(i.first);
            }
        }
        for (auto reader : reader_set) {
            if (opr_set.count(reader) <= 0) {
                return false;
            }
        }
        auto src = rewriter.get_var(conv_bias->input(0)),
             filter = rewriter.get_var(conv_bias->input(1)),
             bias = rewriter.get_var(conv_bias->input(2));
        auto new_bias = ReformatManager::instance().get(
                ReformatKey{TensorFormats::NCHWc32, TensorFormats::NCHWc4})({bias});
        auto new_param = conv_bias->param();
        new_param.format = megdnn::param::ConvBias::Format::NCHW32_NCHW4;
        auto conv_bias_shuffle = opr::ConvBias::make(
                src, filter, new_bias, new_param, conv_bias->execution_policy(),
                conv_bias->config());
        rewriter.replace_var(
                opr->output(0), conv_bias_shuffle.node(),
                mgb_cstr_log("replace conv_bias + "
                             "reformat to conv_bias(NCHW32_NCHW4)"));
        return true;
    };
    MGB_MARK_USED_VAR(try_conv_reformat_nchw322nchw4);
    MGB_MARK_USED_VAR(try_conv_reformat_nchw42nchw32);

    auto on_opr = [&try_conv_dimshuffle_reshape_typecvt,
                   &try_conv_reformat_nchw42nchw32, &try_conv_reformat_nchw42nhwc,
                   &try_conv_reformat_nchw322nchw4, &rewriter](OperatorNodeBase* opr) {
        if (!try_conv_dimshuffle_reshape_typecvt(opr) &&
            !try_conv_reformat_nchw42nchw32(opr) &&
            !try_conv_reformat_nchw42nhwc(opr) &&
            !try_conv_reformat_nchw322nchw4(opr)) {
            rewriter.auto_replace_outputs(opr);
        }
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();

    MIDOUT_E
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
