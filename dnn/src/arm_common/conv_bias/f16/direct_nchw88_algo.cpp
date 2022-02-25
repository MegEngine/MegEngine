#include "megdnn/oprs.h"
#include "src/arm_common/conv_bias/block_helper.h"
#include "src/arm_common/conv_bias/f16/algos.h"
#include "src/arm_common/conv_bias/f16/direct_nchw88_kern.h"

#include "src/arm_common/elemwise_helper/elemwise_op.h"

#include "midout.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

using namespace megdnn;
using namespace arm_common;
using conv_fun = std::function<void(
        const WorkspaceBundle& bundle, const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index, const CpuNDRange& workspace_ids)>;
MIDOUT_DECL(megdnn_arm_common_conv_bias_fp16_nchw88)
namespace {

static WorkspaceBundle get_bundle(const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    size_t nr_threads = param.nr_threads;
    size_t IC = fm.icpg / 8;
    size_t PH = fm.padding[0];
    size_t PW = fm.padding[1];
    size_t IH2 = param.isz[0] + 2 * PH;
    size_t IW2 = param.isz[1] + 2 * PW;
    if (PH == 0 && PW == 0) {
        return {nullptr, {}};
    }

    size_t s = (nr_threads * IC * IH2 * IW2 * 8) * sizeof(dt_float16);
    return {nullptr, {s}};
}

void copy_padding_kern(
        const WorkspaceBundle& bundle, const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index, const CpuNDRange& workspace_ids) {
    auto fm = kern_param.filter_meta;
    size_t group = fm.group;
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t IC = fm.icpg / 8;
    size_t PH = fm.padding[0];
    size_t PW = fm.padding[1];
    size_t IH2 = IH + 2 * PH;
    size_t IW2 = IW + 2 * PW;

    if (PH == 0 && PW == 0) {
        return;
    }

    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0];
    size_t workspace_batch_id = workspace_ids[1];
    size_t channel_id = workspace_ids[2];
    size_t group_id = ncb_index.ndrange_id[0];
    size_t batch_id = ncb_index.ndrange_id[1];

    const dt_float16* sptr =
            kern_param.src<dt_float16>(batch_id, group_id, channel_id, 1, 8);

    //! copy to sptr_base to eliminate padding effect
    dt_float16* sptr_base = static_cast<dt_float16*>(bundle.get(0)) +
                            workspace_batch_id * group * IC * IH2 * IW2 * 8 +
                            workspace_group_id * IC * IH2 * IW2 * 8 +
                            channel_id * IH2 * IW2 * 8;
    std::memset(sptr_base, 0, IH2 * IW2 * 8 * sizeof(dt_float16));
    rep(ih, IH) {
        std::memcpy(
                sptr_base + (ih + PH) * IW2 * 8 + PW * 8, sptr + ih * IW * 8,
                IW * 8 * sizeof(dt_float16));
    }
};

template <size_t FH, size_t SH, BiasMode bias_mode, typename Op>
static void do_conv_kern(
        const WorkspaceBundle& bundle, const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index, const CpuNDRange& workspace_ids) {
    auto fm = kern_param.filter_meta;
    size_t group = fm.group;
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t FW = FH;
    size_t IC = fm.icpg / 8;
    size_t PH = fm.padding[0];
    size_t PW = fm.padding[1];
    size_t IH2 = kern_param.isz[0] + 2 * PH;
    size_t IW2 = kern_param.isz[1] + 2 * PW;

    size_t group_id = ncb_index.ndrange_id[0];
    size_t batch_id = ncb_index.ndrange_id[1];
    size_t channel_id = workspace_ids[2];

    //! Used for get the workspace offset
    size_t workspace_batch_id = workspace_ids[1];
    size_t workspace_group_id = workspace_ids[0];

    const __fp16* sptr = nullptr;
    if (PH == 0 && PW == 0) {
        sptr = reinterpret_cast<const __fp16*>(
                kern_param.src<dt_float16>(batch_id, group_id));
    } else {
        sptr = reinterpret_cast<const __fp16*>(
                       static_cast<const dt_float16*>(bundle.get(0))) +
               workspace_batch_id * group * IC * IH2 * IW2 * 8 +
               workspace_group_id * IC * IH2 * IW2 * 8;
    }
    const __fp16* filter = reinterpret_cast<const __fp16*>(
                                   kern_param.filter<dt_float16>(group_id, 1)) +
                           channel_id * IC * FH * FW * 8 * 8;
    const __fp16* bias_ptr = reinterpret_cast<const __fp16*>(
            kern_param.bias<dt_float16>(batch_id, group_id, channel_id, 1, 8));
    __fp16* dptr = reinterpret_cast<__fp16*>(
            kern_param.dst<dt_float16>(batch_id, group_id, channel_id, 1, 8));

    conv_bias::conv_direct_fp16_nchw88<FH, SH, bias_mode, Op>(
            sptr, filter, bias_ptr, dptr, IC, IH2, IW2, OH, OW);
}

}  // namespace

/* ===================== stride1 algo ===================== */
bool ConvBiasImpl::AlgoF16DirectNCHW88::usable(
        const NCBKernSizeParam& param, AlgoSelectionStrategy) const {
    auto&& fm = param.filter_meta;
    auto fh = fm.spatial[0];
    int oc = fm.ocpg;
    int ic = fm.icpg;
    bool ok_type = ((param.src_type.enumv() == DTypeEnum::Float16 &&
                     param.filter_type.enumv() == DTypeEnum::Float16 &&
                     (param.dst_type.enumv() == DTypeEnum::Float16))) &&
                   (fm.format == param::Convolution::Format::NCHW88);
    bool ok_src_dst = (oc % 8 == 0 && oc >= 8 && ic % 8 == 0 && ic >= 8);
    bool ok_filter = fm.spatial_ndim == 2 && fh == fm.spatial[1] &&
                     (fh == 1 || fh == 2 || fh == 3 || fh == 5 || fh == 7);
    bool ok_slide = fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                    ((fm.stride[0] == 1 && fm.stride[1] == 1) ||
                     (fm.stride[0] == 2 && fm.stride[1] == 2));
    bool ok_conv = !fm.should_flip;
    bool ok_comp = param.compute_mode == Param::ComputeMode::DEFAULT;
    return ok_type && ok_src_dst && ok_filter && ok_slide && ok_conv && ok_comp;
}

size_t ConvBiasImpl::AlgoF16DirectNCHW88::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(
            megdnn_arm_common_conv_bias_fp16_nchw88_stride1,
            midout_iv("AlgoF16DirectNCHW88::get_workspace"_hash)) {
        return get_bundle(param).total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoF16DirectNCHW88::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    size_t batch = param.n;
    size_t group = fm.group;

    WorkspaceBundle wbundle = get_bundle(param);
    conv_fun do_conv_fun = nullptr;
    // NOTE: remain_w is not used to gen hash of midout for compatible with
// shape runtime
#define DO_CONV_KERN_FUN(filter, bias_mode, op, stride)            \
    MIDOUT_BEGIN(                                                  \
            megdnn_arm_common_conv_bias_fp16_nchw88,               \
            midout_iv(#filter #bias_mode #stride #op##_hash)) {    \
        do_conv_fun = do_conv_kern<filter, stride, bias_mode, op>; \
    }                                                              \
    MIDOUT_END();

#define GET_STRIDE_PARAM(filter, bias_mode, op)         \
    switch (fm.stride[0]) {                             \
        case 1:                                         \
            DO_CONV_KERN_FUN(filter, bias_mode, op, 1); \
            break;                                      \
        case 2:                                         \
            DO_CONV_KERN_FUN(filter, bias_mode, op, 2); \
            break;                                      \
                                                        \
        default:                                        \
            megdnn_assert(0, "stride not supported");   \
    }

#define GET_OP_PARAM(filter, bias_mode)                            \
    switch (param.nonlineMode) {                                   \
        case param::ConvBias::NonlineMode::IDENTITY:               \
            GET_STRIDE_PARAM(filter, bias_mode, NoneOp<__fp16>)    \
            break;                                                 \
        case param::ConvBias::NonlineMode::RELU:                   \
            GET_STRIDE_PARAM(filter, bias_mode, ReluOp<__fp16>)    \
            break;                                                 \
        case param::ConvBias::NonlineMode::H_SWISH:                \
            GET_STRIDE_PARAM(filter, bias_mode, HSwishOp<__fp16>)  \
            break;                                                 \
        case param::ConvBias::NonlineMode::SIGMOID:                \
            GET_STRIDE_PARAM(filter, bias_mode, SigmoidOp<__fp16>) \
            break;                                                 \
        default:                                                   \
            megdnn_assert(0, "nonline not supported");             \
            break;                                                 \
    }

#define GET_BIAS_MODE_PARAM(filter)                                \
    switch (param.bias_mode) {                                     \
        case BiasMode::NO_BIAS:                                    \
            GET_OP_PARAM(filter, BiasMode::NO_BIAS)                \
            break;                                                 \
        case BiasMode::BROADCAST_CHANNEL_BIAS:                     \
            GET_OP_PARAM(filter, BiasMode::BROADCAST_CHANNEL_BIAS) \
            break;                                                 \
        case BiasMode::BIAS:                                       \
            GET_OP_PARAM(filter, BiasMode::BIAS)                   \
            break;                                                 \
        default:                                                   \
            megdnn_assert(0, "bias_mode not supported");           \
            break;                                                 \
    }

#define DISPATCH_CONV_KERN()                          \
    switch (param.filter_meta.spatial[0]) {           \
        case 1:                                       \
            GET_BIAS_MODE_PARAM(1)                    \
            break;                                    \
        case 2:                                       \
            GET_BIAS_MODE_PARAM(2)                    \
            break;                                    \
        case 3:                                       \
            GET_BIAS_MODE_PARAM(3)                    \
            break;                                    \
        case 5:                                       \
            GET_BIAS_MODE_PARAM(5)                    \
            break;                                    \
        case 7:                                       \
            GET_BIAS_MODE_PARAM(7)                    \
            break;                                    \
        default:                                      \
            megdnn_assert(0, "filter not supported"); \
            break;                                    \
    }

    DISPATCH_CONV_KERN();

#undef DO_CONV_KERN_FUN
#undef GET_REMAIN_W_PARAM
#undef GET_OP_PARAM
#undef GET_BIAS_MODE_PARAM
#undef DISPATCH_CONV_KERN

    megdnn_assert(do_conv_fun);

    WorkspaceBundle bundle = get_bundle(param);

    SmallVector<ConvBiasImpl::NCBKern> ret_kerns;

    auto exec_one_group = [bundle, do_conv_fun](
                                  const NCBKernParam& kern_param,
                                  const NCBKernIndex& ncb_index) mutable {
        auto fm = kern_param.filter_meta;
        size_t IC = fm.icpg / 8;
        size_t OC = fm.ocpg / 8;
        bundle.set(kern_param.workspace_ptr);
        for (size_t ic = 0; ic < IC; ic++) {
            copy_padding_kern(
                    bundle, kern_param, ncb_index, {ncb_index.thread_id, 0, ic});
        }
        for (size_t oc = 0; oc < OC; oc++) {
            do_conv_fun(bundle, kern_param, ncb_index, {ncb_index.thread_id, 0, oc});
        }
    };
    // TODO: large group only, further multithread optimization required
    ret_kerns.push_back({exec_one_group, {group, batch, 1_z}});

    return ret_kerns;
}

#endif

// vim: syntax=cpp.doxygen
