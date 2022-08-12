#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "src/common/nchw_nchwxx_valid.h"
#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/gi/fp32/algos.h"
#include "src/fallback/elemwise_helper/elemwise_op.h"

#include "midout.h"

using namespace megdnn;
using namespace fallback;

MIDOUT_DECL(megdnn_fallback_conv_bias_fp32_nchw_nchw44_agent)

namespace {

param::ConvBias get_param_convbias(const ConvBiasImpl::NCBKernSizeParam& p) {
    param::ConvBias::Mode mode;
    if (p.filter_meta.should_flip) {
        mode = param::ConvBias::Mode::CONVOLUTION;
    } else {
        mode = param::ConvBias::Mode::CROSS_CORRELATION;
    }

    return param::ConvBias{
            p.nonlineMode,
            mode,
            param::ConvBias::Sparse::DENSE,
            ConvBias::Param::Format::NCHW,
            p.filter_meta.padding[0],
            p.filter_meta.padding[1],
            p.filter_meta.stride[0],
            p.filter_meta.stride[1],
            p.filter_meta.dilation[0],
            p.filter_meta.dilation[1],
            megdnn::param::ConvBias::ComputeMode::DEFAULT};
}

TensorLayoutArray get_layouts(const ConvBiasImpl::NCBKernSizeParam& p) {
    UNPACK_CONV_NCB_KERN_SIZES(p);
    MEGDNN_MARK_USED_VAR(SH);
    MEGDNN_MARK_USED_VAR(SW);
    MEGDNN_MARK_USED_VAR(PH);
    MEGDNN_MARK_USED_VAR(PW);
    MEGDNN_MARK_USED_VAR(OW);
    MEGDNN_MARK_USED_VAR(OH);
    TensorLayout src_layout({N, IC, IH, IW}, p.src_type);
    //! 44 filter to chw
    TensorLayout filter_layout44({OC / 4, FH, FW, IC, 4}, p.filter_type);
    TensorLayout filter_layout_reshape({OC / 4, 4, IC, FH, FW}, p.filter_type);
    TensorLayout filter_layout({OC, IC, FH, FW}, p.filter_type);

    TensorLayout bias_layout44{{}, p.bias_type};
    TensorLayout bias_layout{{}, p.bias_type};
    TensorLayout bias_layout_reshape{{}, p.bias_type};
    if (p.bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_layout44 = TensorLayout({1, OC / 4, 1, 1, 4}, p.bias_type);
        bias_layout_reshape = TensorLayout({1, OC / 4, 4, 1, 1}, p.bias_type);
        bias_layout = TensorLayout({1, OC, 1, 1}, p.bias_type);
    }
    //! chw dst to 44
    TensorLayout dst_layout = TensorLayout({N, OC, OH, OW}, p.dst_type);
    TensorLayout dst_layout_reshape = TensorLayout({N, OC / 4, 4, OH, OW}, p.dst_type);
    TensorLayout dst_layout44 = TensorLayout({N, OC / 4, OH, OW, 4}, p.dst_type);

    return {src_layout,        filter_layout,         filter_layout44,
            bias_layout,       bias_layout44,         dst_layout,
            dst_layout44,      filter_layout_reshape, bias_layout_reshape,
            dst_layout_reshape};
}

static WorkspaceBundle get_bundle(
        const ConvBiasImpl::NCBKernSizeParam& param,
        const std::unique_ptr<ConvBias>& conv_bias_op) {
    auto layouts = get_layouts(param);
    auto src_layout = layouts[0];
    auto filter_layout = layouts[1];
    auto bias_layout = layouts[3];
    auto dst_layout = layouts[5];
    size_t weight_relayout_workspace = filter_layout.span().dist_byte();
    size_t bias_relayout_workspace = bias_layout.span().dist_byte();
    conv_bias_op->param() = get_param_convbias(param);
    auto dummy = TensorLayout();
    auto conv_workspace = conv_bias_op->get_workspace_in_bytes(
            src_layout, filter_layout, bias_layout, dummy, dst_layout, nullptr);
    auto conv_dst_workspace = dst_layout.span().dist_byte();

    return {nullptr,
            {weight_relayout_workspace, bias_relayout_workspace, conv_workspace,
             conv_dst_workspace}};
};
};  // namespace

namespace {
inline bool is_usable(
        const DTypeEnum src_dtype, const DTypeEnum filter_dtype,
        const DTypeEnum dst_dtype,
        const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& fm,
        const BiasMode bias_mode, const param::ConvBias::NonlineMode nonline_mode) {
    bool ok_type =
            ((src_dtype == DTypeEnum::Float32 && filter_dtype == DTypeEnum::Float32 &&
              (dst_dtype == DTypeEnum::Float32))) &&
            (fm.format == param::Convolution::Format::NCHW44);
    bool ok_nonline = nonline_mode == param::ConvBias::NonlineMode::IDENTITY ||
                      nonline_mode == param::ConvBias::NonlineMode::RELU ||
                      nonline_mode == param::ConvBias::NonlineMode::SIGMOID ||
                      nonline_mode == param::ConvBias::NonlineMode::H_SWISH;
    bool ok_src_dst =
            fm.icpg < 4 && (fm.ocpg % 4 == 0 && fm.ocpg >= 4) && fm.group == 1;

    bool ok_filter = fm.spatial_ndim == 2 && fm.spatial[0] == fm.spatial[1] &&
                     (fm.spatial[0] == 2 || fm.spatial[0] == 3 || fm.spatial[0] == 5 ||
                      fm.spatial[0] == 7);
    bool ok_slide = fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                    fm.stride[0] == fm.stride[1] &&
                    (fm.stride[0] == 1 || fm.stride[1] == 2);
    bool ok_conv = !fm.should_flip && bias_mode != BiasMode::BIAS;
    bool avaible =
            ok_type && ok_nonline && ok_src_dst && ok_filter && ok_slide && ok_conv;
    return avaible;
}
};  // namespace

bool ConvBiasImpl::AlgoF32DirectNCHWNCHW44AGENT::usable(
        const NCBKernSizeParam& param, AlgoSelectionStrategy) const {
    return is_usable(
            param.src_type.enumv(), param.filter_type.enumv(), param.dst_type.enumv(),
            param.filter_meta, param.bias_mode, param.nonlineMode);
}

size_t ConvBiasImpl::AlgoF32DirectNCHWNCHW44AGENT::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(
            megdnn_fallback_conv_bias_fp32_nchw_nchw44_agent,
            midout_iv("AlgoF32DirectNCHWNCHW44AGENT::get_workspace"_hash)) {
        auto conv_bias_op = param.handle->create_operator<ConvBias>();
        return get_bundle(param, conv_bias_op).total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoF32DirectNCHWNCHW44AGENT::
        dispatch_kerns(const NCBKernSizeParam& k_param) const {
    SmallVector<ConvBiasImpl::NCBKern> ret_kerns;

    MIDOUT_BEGIN(
            megdnn_fallback_conv_bias_fp32_nchw_nchw44_agent,
            midout_iv("AlgoF32DirectNCHWNCHW44AGENT::dispatch_kerns"_hash)) {
        auto filter_and_bias_dimshuffle = [](const NCBKernParam& kern_param,
                                             const NCBKernIndex&) {
            auto layouts = get_layouts(kern_param);
            auto filter_layout_44 = layouts[2];
            auto bias_layout44 = layouts[4];
            auto filter_layout_reshape = layouts[7];
            auto bias_layout_reshape = layouts[8];

            auto conv_bias_op = kern_param.handle->create_operator<ConvBias>();
            auto bundle = get_bundle(kern_param, conv_bias_op);
            bundle.set(kern_param.workspace_ptr);
            auto weight_ws = bundle.get(0);
            auto bias_ws = bundle.get(1);

            //! relayout bias and weight
            TensorND chw_weight_t = TensorND(weight_ws, filter_layout_reshape);
            TensorND weight44_t = TensorND(
                    kern_param.filter_ptr.get_ptr(),
                    filter_layout_44.dimshuffle({0, 4, 3, 1, 2}));
            auto relayout_op = inplace_cpu_handle()->create_operator<Relayout>();
            relayout_op->exec(weight44_t, chw_weight_t);

            TensorND chw_bias_t = TensorND(bias_ws, bias_layout_reshape);
            if (bias_layout44.ndim != 0) {
                TensorND bias44_t = TensorND(
                        kern_param.bias_ptr.get_ptr(),
                        bias_layout44.dimshuffle({0, 1, 4, 2, 3}));
                relayout_op->exec(bias44_t, chw_bias_t);
            }
        };
        ret_kerns.push_back({filter_and_bias_dimshuffle, {1}});

        auto do_agent_conv = [&ret_kerns, &k_param]() {
            auto layouts = get_layouts(k_param);
            auto src_layout = layouts[0];
            auto filter_layout = layouts[1];
            auto bias_layout = layouts[3];
            auto dst_layout = layouts[5];

            //! do chw conv
            auto conv_bias_op = k_param.handle->create_operator<ConvBias>();
            conv_bias_op->param() = get_param_convbias(k_param);
            auto dummy_z = TensorND();
            auto&& conv_bias_algo =
                    static_cast<ConvBiasImpl*>(conv_bias_op.get())
                            ->get_algorithm_heuristic(
                                    src_layout, filter_layout, bias_layout,
                                    dummy_z.layout, dst_layout,
                                    std::numeric_limits<size_t>::max(),
                                    AlgoAttribute::DEFAULT, AlgoAttribute::DEFAULT);
            auto new_param = k_param;
            new_param.filter_meta.format = ConvBias::Param::Format::NCHW;
            auto&& conv_bias_kerns =
                    static_cast<AlgoBase*>(conv_bias_algo)->dispatch_kerns(new_param);
            for (size_t i = 0; i < conv_bias_kerns.size(); i++) {
                auto&& kernel = conv_bias_kerns[i];
                auto run = [kernel](
                                   const NCBKernParam& p,
                                   const NCBKernIndex& ncb_index) {
                    auto conv_bias_op = p.handle->create_operator<ConvBias>();
                    auto bundle = get_bundle(p, conv_bias_op);
                    bundle.set(p.workspace_ptr);
                    auto weight_ws = bundle.get(0);
                    auto bias_ws = bundle.get(1);
                    auto chw_conv_ws = bundle.get(2);
                    auto chw_conv_ws_size = bundle.get_size(2);
                    auto chw_conv_dst_ws = bundle.get(3);

                    auto param = p;
                    param.filter_ptr = weight_ws;
                    param.bias_ptr = bias_ws;
                    param.dst_ptr = chw_conv_dst_ws;
                    param.workspace_ptr = chw_conv_ws;
                    param.workspace_size = chw_conv_ws_size;
                    kernel.kern(param, {ncb_index.thread_id, ncb_index.ndrange_id});
                };
                ret_kerns.push_back({run, kernel.global_size});
            }
        };
        do_agent_conv();

        auto dest_dimshuffle = [](const NCBKernParam& kern_param, const NCBKernIndex&) {
            auto param = kern_param;
            auto layouts = get_layouts(param);
            auto dst_layout44 = layouts[6];
            auto dst_layout_reshape = layouts[9];

            auto conv_bias_op = kern_param.handle->create_operator<ConvBias>();
            auto bundle = get_bundle(kern_param, conv_bias_op);
            bundle.set(kern_param.workspace_ptr);
            auto chw_conv_dst_ws = bundle.get(3);

            //! relayout dst to dst44 tensor
            TensorND chw44_dst_t = TensorND(kern_param.dst_ptr.get_ptr(), dst_layout44);
            TensorND chw_dst_t = TensorND(chw_conv_dst_ws, dst_layout_reshape);
            auto relayout_op = inplace_cpu_handle()->create_operator<Relayout>();
            relayout_op->exec(
                    {chw_conv_dst_ws, dst_layout_reshape.dimshuffle({0, 1, 3, 4, 2})},
                    chw44_dst_t);
        };
        ret_kerns.push_back({dest_dimshuffle, {1}});
        return ret_kerns;
    }
    MIDOUT_END();
}

// vim: syntax=cpp.doxygen
