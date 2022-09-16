#include "megbrain/gopt/inference.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/misc.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/opr_shallow_copy.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/tensor_format.h"

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "megbrain/gopt/misc.h"
#include "megbrain/utils/hash_ct.h"

#include "midout.h"

#include "megbrain/gopt/reformat_manager.h"

MIDOUT_DECL(megbrain_padding_channel)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_padding_channel, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace gopt;
using ReformatKey = ReformatManager::ReformatKey;

/* ==================== PaddingChannelPass ================= */
namespace {
size_t padding_int4(size_t in_channel, bool flag) {
    static_cast<void>(flag);
    if (in_channel <= 32) {
        return (8 - (in_channel % 8)) % 8;
    } else {
        return (64 - (in_channel % 64)) % 64;
    }
}

size_t padding_int8(size_t in_channel, bool flag) {
    if (flag) {
        if (in_channel <= 16) {
            return (4 - (in_channel % 4)) % 4;
        } else {
            return (32 - (in_channel % 32)) % 32;
        }
    } else {
        return (4 - (in_channel % 4)) % 4;
    }
}
size_t padding_4(size_t in_channel, bool) {
    return (4 - (in_channel % 4)) % 4;
};

}  // namespace

std::unique_ptr<PaddingChannelPass> PaddingChannelPass::make(
        cg::GraphCommonOptimizeOptions::LayoutTransform layout_transform) {
    MIDOUT_B("PaddingChannelPass::make")
    using LayoutTrans = cg::GraphCommonOptimizeOptions::LayoutTransform;
    auto ret = std::make_unique<PaddingChannelPass>();
    auto& alignment_map = ret->m_alignment_map;
    if (layout_transform == LayoutTrans::NCHW64) {
        alignment_map[DTypeEnum::QuantizedS4] = padding_int4;
        alignment_map[DTypeEnum::Quantized4Asymm] = padding_int4;
        alignment_map[DTypeEnum::QuantizedS8] = padding_int8;
    } else if (
            layout_transform == LayoutTrans::NCHW44 ||
            layout_transform == LayoutTrans::NCHW44_DOT) {
        alignment_map[DTypeEnum::QuantizedS8] = padding_4;
        alignment_map[DTypeEnum::Quantized8Asymm] = padding_4;
        alignment_map[DTypeEnum::Float32] = padding_4;
    }
    ret->fill_opr_convert_fun(layout_transform);
    return ret;
    MIDOUT_E
}
const char* PaddingChannelPass::name() const {
    return mgb_cstr_log("padding output channel to multiple of 4/32");
}

void PaddingChannelPass::apply(OptState& opt) const {
    MIDOUT_B("PaddingChannelPass::apply");
    // do not check shape
    opt.set_var_replace_check_flag(
            VarReplaceCheckFlag::CHECK_ALL ^ VarReplaceCheckFlag::CHECK_SHAPE);
    m_padding_oprs.clear();
    auto rewriter = opt.graph().make_rewriter();
    auto on_opr = [this, &opt, &rewriter](OperatorNodeBase* opr) {
        auto it = m_opr_replace_funcs.find(opr->dyn_typeinfo());
        if (it != m_opr_replace_funcs.end()) {
            VarNodeArray new_inp;
            new_inp.reserve(opr->input().size());
            for (auto&& inp : opr->input()) {
                new_inp.push_back(rewriter.get_var(inp));
            }
            auto new_opr = (it->second)(opr, new_inp);
            auto &&out0 = opr->output(), &&out1 = new_opr->output();
            mgb_assert(
                    out0.size() == out1.size(),
                    "bad opr replace: src=%s{%s} dst=%s{%s}, "
                    "src.size=%zu "
                    "dst.size=%zu",
                    opr->cname(), opr->dyn_typeinfo()->name, new_opr->cname(),
                    new_opr->dyn_typeinfo()->name, out0.size(), out1.size());
            for (size_t i = 0; i < out0.size(); ++i) {
                if (!out0[i]->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                    mgb_assert(!out1[i]->contain_flag(VarNode::Flag::VOLATILE_CONTENT));
                    auto src = out0[i];
                    auto dst = out1[i];
                    if (opt.graph().endpoint_contain(src) &&
                        !src->shape().eq_shape(dst->shape())) {
                        dst = extract_subtensor(dst, src->shape());
                    }
                    rewriter.replace_var(src, dst, nullptr);
                }
            }
        } else {
            rewriter.auto_replace_outputs(opr);
        }
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();

    MIDOUT_E
}

VarNode* PaddingChannelPass::extract_subtensor(
        VarNode* inp, const TensorShape& orig_shape) const {
    mgb_assert(inp->shape().ndim == 4);
    mgb_assert(inp->shape()[0] == orig_shape[0]);
    mgb_assert(inp->shape()[2] == orig_shape[2]);
    mgb_assert(inp->shape()[3] == orig_shape[3]);
    size_t orig_channels = orig_shape[1];
    auto x = SymbolVar(inp);
    auto cv = [&x](int v) { return x.make_scalar(v); };
    using AIdx = opr::Subtensor::AxisIndexer;
    auto sub = opr::Subtensor::make(
            x, {AIdx::make_interval(0, None, None, cv(1)),
                AIdx::make_interval(1, None, cv(orig_channels), None),
                AIdx::make_interval(2, None, None, cv(1)),
                AIdx::make_interval(3, None, None, cv(1))});
    return sub.node();
};

VarNode* PaddingChannelPass::pad_in_channels(VarNode* inp, size_t pad_channels) {
    mgb_assert(inp->shape().ndim == 4);
    TensorShape shape{inp->shape()[0], pad_channels, inp->shape()[2], inp->shape()[3]};
    std::shared_ptr<HostTensorND> host_val =
            std::make_shared<HostTensorND>(inp->comp_node(), inp->dtype());
    host_val->resize(shape);
    auto ptr = host_val->raw_ptr();
    size_t size_bytes = TensorLayout{shape, inp->dtype()}.span().dist_byte();
    std::memset(ptr, 0, size_bytes);
    auto padding = opr::ImmutableTensor::make(*inp->owner_graph(), *host_val);
    auto out = opr::Concat::make({inp, padding}, 1);
    return out.node();
};

VarNode* PaddingChannelPass::pad_out_channels(VarNode* inp, size_t pad_channels) {
    mgb_assert(inp->shape().ndim == 4);
    TensorShape shape{pad_channels, inp->shape()[1], inp->shape()[2], inp->shape()[3]};
    std::shared_ptr<HostTensorND> host_val =
            std::make_shared<HostTensorND>(inp->comp_node(), inp->dtype());
    host_val->resize(shape);
    auto ptr = host_val->raw_ptr();
    size_t size_bytes = TensorLayout{shape, inp->dtype()}.span().dist_byte();
    std::memset(ptr, 0, size_bytes);
    auto padding = opr::ImmutableTensor::make(*inp->owner_graph(), *host_val);
    auto out = opr::Concat::make({inp, padding}, 0);
    return out.node();
};

// padding policy for conv bias with data type qint8
OperatorNodeBase* PaddingChannelPass::padding_policy(
        OperatorNodeBase* opr, const VarNodeArray& new_inp) {
    mgb_assert(opr->input().size() == new_inp.size());
    mgb_assert(new_inp.size() == 3);
    //! new weights and old weights are same shape
    mgb_assert(opr->input(1)->shape().eq_shape(new_inp[1]->shape()));
    auto inps = new_inp;
    size_t out_channels = opr->input(1)->shape()[0];
    size_t in_channels = opr->input(1)->shape()[1];
    size_t new_in_channels = new_inp[0]->shape()[1];
    auto it = m_alignment_map.find(opr->input(0)->dtype().enumv());
    if (it != m_alignment_map.end()) {
        mgb_assert(it->second);
    } else {
        return serialization::copy_opr_shallow(*opr, inps, opr->config());
    }
    // pad input channels
    if (m_padding_oprs.count(opr->input(0)->owner_opr())) {
        //! as the opr of input var is padding, but the dtype of input and output of
        //! the input opr maybe different, so the alignment is not the same
        size_t pad_channels_0 = it->second(new_in_channels, true);
        size_t pad_channels_1 = it->second(in_channels, true);
        if (pad_channels_0) {
            inps[0] = pad_in_channels(new_inp[0], pad_channels_0);
        } else {
            pad_channels_1 = new_in_channels - in_channels;
        }
        if (pad_channels_1) {
            inps[1] = pad_in_channels(new_inp[1], pad_channels_1);
        }
    } else {
        mgb_assert(new_in_channels == in_channels);
        size_t pad_channels = it->second(in_channels, true);
        if (pad_channels > 0) {
            inps[0] = pad_in_channels(new_inp[0], pad_channels);
            inps[1] = pad_in_channels(new_inp[1], pad_channels);
        }
    }
    out_channels = inps[1]->shape()[0];
    size_t pad_channels = it->second(out_channels, true);
    if (pad_channels > 0) {
        inps[1] = pad_out_channels(inps[1], pad_channels);
        inps[2] = pad_in_channels(inps[2], pad_channels);
        m_padding_oprs.insert(opr);
    }
    return serialization::copy_opr_shallow(*opr, inps, opr->config());
};

void PaddingChannelPass::fill_opr_convert_fun(LayoutTrans layout_trans) {
    add_convbias_replace_func(layout_trans);
    add_conv_backward_data_replace_func(layout_trans);
    add_format_aware_opr_replace_func(layout_trans);
    add_elemwise_like_opr_replace_func(layout_trans);
    add_nonpadding_oprs_replace_func(layout_trans);
}

void PaddingChannelPass::add_convbias_replace_func(LayoutTrans layout_trans) {
    if (layout_trans == LayoutTrans::NCHW64) {
        m_opr_replace_funcs[opr::ConvBiasForward::typeinfo()] =
                [this](OperatorNodeBase* opr, const VarNodeArray& new_inp) {
                    if (opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8) {
                        return padding_policy(opr, new_inp);
                    } else if (
                            opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS4 ||
                            opr->input(0)->dtype().enumv() ==
                                    DTypeEnum::Quantized4Asymm) {
                        return padding_policy(opr, new_inp);
                    } else {
                        mgb_assert(
                                m_padding_oprs.count(opr->input(0)->owner_opr()) == 0,
                                "conv bias operator for data type(%s) cannot be "
                                "padded channel. "
                                "consumer(%s), producer(%s)",
                                opr->input(0)->dtype().name(), opr->cname(),
                                opr->input(0)->owner_opr()->cname());
                        return serialization::copy_opr_shallow(
                                *opr, new_inp, opr->config());
                    }
                };
    } else if (layout_trans == LayoutTrans::NCHW44) {
        m_opr_replace_funcs[opr::ConvBiasForward::typeinfo()] =
                [this](OperatorNodeBase* opr, const VarNodeArray& new_inp) {
                    return padding_policy(opr, new_inp);
                };
    }
}

void PaddingChannelPass::add_conv_backward_data_replace_func(LayoutTrans layout_trans) {
    if (layout_trans == LayoutTrans::NCHW64) {
        m_opr_replace_funcs[opr::ConvolutionBackwardData::typeinfo()] =
                [this](OperatorNodeBase* opr, const VarNodeArray& new_inp) {
                    if (opr->input(1)->dtype().enumv() != DTypeEnum::QuantizedS8) {
                        mgb_assert(
                                m_padding_oprs.count(opr->input(0)->owner_opr()) == 0,
                                "conv bwd data operator for data type(%s) cannot "
                                "be "
                                "padded channel. "
                                "consumer(%s), producer(%s)",
                                opr->input(0)->dtype().name(), opr->cname(),
                                opr->input(0)->owner_opr()->cname());
                        return serialization::copy_opr_shallow(
                                *opr, new_inp, opr->config());
                    }
                    mgb_assert(opr->input().size() == new_inp.size());
                    mgb_assert(
                            new_inp.size() == 2,
                            "deconv (conv bwd data) operator for inference can "
                            "only have 2 input vars(got:%zu)",
                            new_inp.size());
                    mgb_assert(opr->input(0)->shape().eq_shape(new_inp[0]->shape()));
                    auto inps = new_inp;
                    size_t out_channels = opr->input(0)->shape()[0];
                    size_t in_channels = opr->input(0)->shape()[1];
                    size_t new_out_channels = new_inp[1]->shape()[1];
                    auto it = m_alignment_map.find(opr->input(1)->dtype().enumv());
                    // pad output channels
                    if (m_padding_oprs.count(opr->input(1)->owner_opr())) {
                        size_t pad_channels = new_out_channels - out_channels;
                        inps[0] = pad_out_channels(new_inp[0], pad_channels);
                    } else {
                        size_t pad_channels = it->second(out_channels, false);
                        if (pad_channels > 0) {
                            inps[0] = pad_out_channels(new_inp[0], pad_channels);
                            inps[1] = pad_in_channels(new_inp[1], pad_channels);
                        }
                    }
                    out_channels = inps[0]->shape()[0];
                    // pad input channels
                    size_t pad_channels = it->second(in_channels, false);
                    if (pad_channels > 0) {
                        inps[0] = pad_in_channels(inps[0], pad_channels);
                        m_padding_oprs.insert(opr);
                    }
                    return serialization::copy_opr_shallow(*opr, inps, opr->config());
                };
    }
}

void PaddingChannelPass::add_format_aware_opr_replace_func(LayoutTrans) {
    auto replace_format_aware_opr = [this](OperatorNodeBase* opr,
                                           const VarNodeArray& new_inp) {
        if (opr->input(0)->dtype().enumv() != DTypeEnum::QuantizedS8 &&
            opr->input(0)->dtype().enumv() != DTypeEnum::QuantizedS4 &&
            opr->input(0)->dtype().enumv() != DTypeEnum::Quantized4Asymm) {
            mgb_assert(
                    m_padding_oprs.count(opr->input(0)->owner_opr()) == 0,
                    "operator(type:%s,name:%s) for data type(%s) cannot be "
                    "padded channel. extra info:"
                    "consumer(%s), producer(%s)",
                    opr->dyn_typeinfo()->name, opr->cname(),
                    opr->input(0)->dtype().name(), opr->cname(),
                    opr->input(0)->owner_opr()->cname());
            return serialization::copy_opr_shallow(*opr, new_inp, opr->config());
        }
        mgb_assert(opr->input().size() == new_inp.size());
        if (m_padding_oprs.count(opr->input(0)->owner_opr())) {
            m_padding_oprs.insert(opr);
        }
        return serialization::copy_opr_shallow(*opr, new_inp, opr->config());
    };
    m_opr_replace_funcs[opr::PoolingForward::typeinfo()] = replace_format_aware_opr;
    m_opr_replace_funcs[opr::WarpPerspectiveForward::typeinfo()] =
            replace_format_aware_opr;
}

void PaddingChannelPass::add_elemwise_like_opr_replace_func(LayoutTrans) {
    auto replace_elemwise_like_opr = [this](OperatorNodeBase* opr,
                                            const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        bool have_padding_inp = false;
        bool padding_all_inps = true;
        bool same_padding = true;
        size_t channels_after_padding = 0;
        size_t i = 0;
        for (auto&& cur_inp : opr->input()) {
            bool padding_cur_inp = m_padding_oprs.count(cur_inp->owner_opr()) > 0;
            if (padding_cur_inp) {
                if (!have_padding_inp)
                    have_padding_inp = true;
                if (channels_after_padding == 0) {
                    channels_after_padding = new_inp[i]->shape()[1];
                } else {
                    same_padding = channels_after_padding == new_inp[i]->shape()[1];
                }
            }
            if (padding_all_inps && (!padding_cur_inp || !same_padding))
                padding_all_inps = false;
            ++i;
        }
        if (have_padding_inp && !padding_all_inps) {
            auto inps = new_inp;
            for (size_t i = 0; i < new_inp.size(); ++i) {
                auto cur_inp = opr->input(i);
                bool padding_cur_inp = m_padding_oprs.count(cur_inp->owner_opr()) > 0;
                if (padding_cur_inp) {
                    inps[i] = extract_subtensor(inps[i], cur_inp->shape());
                }
            }
            return serialization::copy_opr_shallow(*opr, inps, opr->config());
        }
        if (padding_all_inps) {
            m_padding_oprs.insert(opr);
        }
        return serialization::copy_opr_shallow(*opr, new_inp, opr->config());
    };
    m_opr_replace_funcs[opr::ElemwiseMultiType::typeinfo()] = replace_elemwise_like_opr;
    m_opr_replace_funcs[opr::Elemwise::typeinfo()] = replace_elemwise_like_opr;
    m_opr_replace_funcs[opr::TypeCvt::typeinfo()] = replace_elemwise_like_opr;
}

void PaddingChannelPass::add_nonpadding_oprs_replace_func(LayoutTrans) {
    auto replace_nonpadding_oprs = [this](OperatorNodeBase* opr,
                                          const VarNodeArray& new_inp) {
        mgb_assert(opr->input().size() == new_inp.size());
        auto inps = new_inp;
        for (size_t i = 0; i < new_inp.size(); ++i) {
            auto cur_inp = opr->input(i);
            bool padding_cur_inp = m_padding_oprs.count(cur_inp->owner_opr()) > 0;
            if (padding_cur_inp) {
                inps[i] = extract_subtensor(inps[i], cur_inp->shape());
            }
        }
        return serialization::copy_opr_shallow(*opr, inps, opr->config());
    };
    m_opr_replace_funcs[opr::Reshape::typeinfo()] = replace_nonpadding_oprs;
    m_opr_replace_funcs[opr::GetVarShape::typeinfo()] = replace_nonpadding_oprs;
    m_opr_replace_funcs[opr::Concat::typeinfo()] = replace_nonpadding_oprs;
    m_opr_replace_funcs[opr::Reduce::typeinfo()] = replace_nonpadding_oprs;
    m_opr_replace_funcs[opr::Subtensor::typeinfo()] = replace_nonpadding_oprs;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
