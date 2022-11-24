#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "src/arm_common/conv_bias/f16/algos.h"
#include "src/arm_common/conv_bias/f16/direct_nchw_nchw88_kern.h"
#include "src/arm_common/elemwise_helper/elemwise_op.h"
#include "src/common/nchw_nchwxx_valid.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/gi/block_helper.h"
#include "src/fallback/conv_bias/opr_impl.h"
using namespace megdnn;
using namespace arm_common;

MIDOUT_DECL(megdnn_arm_common_direct_conv_nchw_nchw88_fp16)

namespace {

static void get_rectified_size(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param, int& ih2,
        int& iw2, int& oh2, int& ow2, int& block_oh) {
    int ic = param.filter_meta.icpg;
    int iw = param.isz[1];
    int oh = param.osz[0];
    int ow = param.osz[1];

    oh2 = oh;
    ow2 = ow;
    iw2 = iw + 2 * static_cast<int>(param.filter_meta.padding[1]);
    const int sh = static_cast<int>(param.filter_meta.stride[0]);
    block_oh = l2_block_helper(param.nr_threads, oh, ic * iw2 * sh * sizeof(__fp16));
    const int fh = static_cast<int>(param.filter_meta.spatial[0]);
    ih2 = (block_oh - 1) * sh + fh;
}

static WorkspaceBundle get_bundle(
        const fallback::ConvBiasImpl::NCBKernSizeParam& param) {
    const auto& fm = param.filter_meta;
    const int group = fm.group;
    const int ic = fm.icpg;
    const int oc = fm.ocpg;
    const int fh = fm.spatial[0];
    const int fw = fm.spatial[1];
    int ih2, iw2, oh2, ow2, oh_block;
    get_rectified_size(param, ih2, iw2, oh2, ow2, oh_block);

    megdnn_assert(oh_block != 0, "oh_block == 0");
    const size_t src_size = ic * ih2 * iw2 * sizeof(__fp16);
    const size_t weight_size = group * oc * ic * fh * fw * sizeof(__fp16);
    return {nullptr, {src_size * param.nr_threads, weight_size}};
}

static void pack_weight(
        const WorkspaceBundle& bundle,
        const fallback::ConvBiasImpl::NCBKernParam& kern_param,
        const fallback::ConvBiasImpl::NCBKernIndex& ncb_index) {
    const int group_id = ncb_index.ndrange_id[0];
    const auto& fm = kern_param.filter_meta;
    const int oc = fm.ocpg;
    const int ic = fm.icpg;
    const int fh = fm.spatial[0];
    const int fw = fm.spatial[1];

    const int oc_idx = 0;
    const int oc_block = oc;

    const __fp16* weight =
            reinterpret_cast<const __fp16*>(kern_param.filter<dt_float16>(group_id)) +
            oc_idx * ic * fh * fw;
    __fp16* packed_weight = static_cast<__fp16*>(bundle.get(1)) +
                            group_id * oc * ic * fh * fw + oc_idx * ic * fh * fw;
    fp16_direct_nchw_nchw88::pack_weight_fp16_nchw_nchw88(
            weight, packed_weight, oc_block, fh, fw, ic);
}

/**
 * @brief Copy data from sptr_origin to sptr, and padding.
 *
 */
static inline void src_copy_pad(
        __fp16* sptr, const __fp16* sptr_origin, const int pw, const int pad_right,
        const int pad_top, const int pad_buttom, const int ic, const int ih,
        const int iw, const int iw2, const int ld_src_ic) {
    rep(ic_idx, ic) {
        const __fp16* ic_sptr_origin = sptr_origin + ic_idx * ld_src_ic;

        memset(sptr, 0, iw2 * pad_top * sizeof(__fp16));
        sptr += iw2 * pad_top;

        rep(ih_idx, ih) {
            memset(sptr, 0, pw * sizeof(__fp16));
            sptr += pw;

            memcpy(sptr, ic_sptr_origin, iw * sizeof(__fp16));
            sptr += iw;
            ic_sptr_origin += iw;

            memset(sptr, 0, pad_right * sizeof(__fp16));
            sptr += pad_right;
        }

        memset(sptr, 0, iw2 * pad_buttom * sizeof(__fp16));
        sptr += iw2 * pad_buttom;
    }
}

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
static void do_conv_kern(
        const WorkspaceBundle& bundle, const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index) {
    const int oc = kern_param.filter_meta.ocpg;
    const int oh = kern_param.osz[0];
    const int ow = kern_param.osz[1];

    const int ic = kern_param.filter_meta.icpg;
    const int ih = kern_param.isz[0];
    const int iw = kern_param.isz[1];

    const int fh = kern_param.filter_meta.spatial[0];
    const int fw = kern_param.filter_meta.spatial[1];

    const int sh = kern_param.filter_meta.stride[0];

    const int ph = kern_param.filter_meta.padding[0];
    const int pw = kern_param.filter_meta.padding[1];

    int ih2, iw2, oh2, ow2, oh_block;
    get_rectified_size(kern_param, ih2, iw2, oh2, ow2, oh_block);

    constexpr int pack_oc = 8;
    const int batch_id = ncb_index.ndrange_id[0];
    const int group_id = ncb_index.ndrange_id[1];
    int oc_idx = 0;
    int oc_block = oc;
    const int oh_idx = ncb_index.ndrange_id[2];
    const int oh_block_real = std::min(oh - oh_idx * oh_block, oh_block);
    const int ih_block_real = (oh_block_real - 1) * sh + fh;
    const int src_top_pad = std::max(0, ph - oh_idx * oh_block * sh);
    const int src_buttom_pad =
            std::max(0, (oh_idx * oh_block + oh_block_real - 1) * sh + fh - ph - ih);
    const int src_right_pad = std::max(iw2 - iw - pw, 0);
    const int src_offset = std::max(oh_idx * oh_block * sh - ph, 0) * iw;
    const __fp16* origin_ptr = reinterpret_cast<const __fp16*>(
                                       kern_param.src<dt_float16>(batch_id, group_id)) +
                               src_offset;
    const size_t src_size = sizeof(__fp16) * ic * ih2 * iw2;
    __fp16* sptr = reinterpret_cast<__fp16*>(
            reinterpret_cast<int8_t*>(bundle.get(0)) + ncb_index.thread_id * src_size);
    src_copy_pad(
            sptr, origin_ptr, pw, src_right_pad, src_top_pad, src_buttom_pad, ic,
            ih_block_real - src_top_pad - src_buttom_pad, iw, iw2, ih * iw);

    //! packed weight
    const __fp16* weight = reinterpret_cast<const __fp16*>(bundle.get(1)) +
                           group_id * oc * ic * fh * fw + oc_idx * ic * fh * fw;
    __fp16* dst =
            reinterpret_cast<__fp16*>(kern_param.dst<dt_float16>(batch_id, group_id)) +
            oh_idx * oh_block * ow * pack_oc;
    const __fp16* bias = reinterpret_cast<const __fp16*>(
                                 kern_param.bias<dt_float16>(batch_id, group_id)) +
                         oc_idx;
    Op op;
    fp16_direct_nchw_nchw88::fp16_direct_conv_nchw_nchw88<
            bias_mode, Op, filter_size, stride>(
            sptr, weight, bias, dst, oc_block, ic, ih_block_real, iw2, oh,
            oh_block_real, ow2, op);
}
}  // namespace

bool ConvBiasImpl::AlgoF16DirectNchwNchw88::usable(
        const NCBKernSizeParam& param, AlgoSelectionStrategy) const {
    return nchw_nchwxx_valid<NchwNchwxxType::NCHW88_FP16>(
            param.src_type.enumv(), param.filter_type.enumv(), param.dst_type.enumv(),
            param.filter_meta, param.bias_mode, param.nonlineMode);
}

size_t ConvBiasImpl::AlgoF16DirectNchwNchw88::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(
            megdnn_arm_common_direct_conv_nchw_nchw88_fp16,
            midout_iv("AlgoF16DirectNchwNchw88::get_workspace"_hash)) {
        return get_bundle(param).total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoF16DirectNchwNchw88::
        dispatch_kerns(const NCBKernSizeParam& param) const {
    using conv_func_ptr = std::function<void(
            const WorkspaceBundle& bundle,
            const fallback::ConvBiasImpl::NCBKernParam& kern_param,
            const fallback::ConvBiasImpl::NCBKernIndex& ncb_index)>;
    const auto& fm = param.filter_meta;
    const int batch = param.n;
    const int group = fm.group;
    auto bundle = get_bundle(param);
    conv_func_ptr conv_func = nullptr;

#define CONV_FUNC(bias_mode, op, filter, stride)                 \
    MIDOUT_BEGIN(                                                \
            megdnn_arm_common_direct_conv_nchw_nchw88_fp16,      \
            midout_iv(#bias_mode #op #filter #stride##_hash)) {  \
        conv_func = do_conv_kern<bias_mode, op, filter, stride>; \
    }                                                            \
    MIDOUT_END();

#define FOR_OP(bias_mode, filter, stride)                           \
    switch (param.nonlineMode) {                                    \
        case param::ConvBias::NonlineMode::IDENTITY:                \
            CONV_FUNC(bias_mode, NoneOp<__fp16>, filter, stride);   \
            break;                                                  \
        case param::ConvBias::NonlineMode::RELU:                    \
            CONV_FUNC(bias_mode, ReluOp<__fp16>, filter, stride);   \
            break;                                                  \
        case param::ConvBias::NonlineMode::H_SWISH:                 \
            CONV_FUNC(bias_mode, HSwishOp<__fp16>, filter, stride); \
            break;                                                  \
        default:                                                    \
            megdnn_assert(0);                                       \
            break;                                                  \
    }

#define FOR_BIAS_MODE(filter, stride)                                 \
    switch (param.bias_mode) {                                        \
        case BiasMode::NO_BIAS:                                       \
            FOR_OP(BiasMode::NO_BIAS, filter, stride);                \
            break;                                                    \
        case BiasMode::BROADCAST_CHANNEL_BIAS:                        \
            FOR_OP(BiasMode::BROADCAST_CHANNEL_BIAS, filter, stride); \
            break;                                                    \
        default:                                                      \
            megdnn_assert(0);                                         \
            break;                                                    \
    }

#define FOR_FILTER(stride)            \
    switch (fm.spatial[0]) {          \
        case 2:                       \
            FOR_BIAS_MODE(2, stride); \
            break;                    \
        case 3:                       \
            FOR_BIAS_MODE(3, stride); \
            break;                    \
        case 5:                       \
            FOR_BIAS_MODE(5, stride); \
            break;                    \
        case 7:                       \
            FOR_BIAS_MODE(7, stride); \
            break;                    \
        default:                      \
            megdnn_assert(0);         \
            break;                    \
    }

#define FOR_STRIDE()          \
    switch (fm.stride[0]) {   \
        case 1:               \
            FOR_FILTER(1);    \
            break;            \
        case 2:               \
            FOR_FILTER(2);    \
            break;            \
        default:              \
            megdnn_assert(0); \
            break;            \
    }

    FOR_STRIDE()

#undef FOR_STRIDE
#undef FOR_FILTER
#undef FOR_BIAS_MODE
#undef FOR_OP
#undef CONV_FUNC

    megdnn_assert(conv_func);

    SmallVector<NCBKern> ret_kerns;
    int oh = param.osz[0];

    int ih2, iw2, oh2, ow2, oh_block;
    get_rectified_size(param, ih2, iw2, oh2, ow2, oh_block);
    auto do_pack_weight = [bundle](
                                  const ConvBiasImpl::NCBKernParam& kern_param,
                                  const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
        bundle.set(kern_param.workspace_ptr);
        pack_weight(bundle, kern_param, ncb_index);
    };
    ret_kerns.push_back({do_pack_weight, {static_cast<size_t>(group)}});

    CpuNDRange ncb_range{
            static_cast<size_t>(batch), static_cast<size_t>(group),
            static_cast<size_t>(div_ceil(oh, oh_block))};
    auto do_conv = [bundle, conv_func](
                           const ConvBiasImpl::NCBKernParam& kern_param,
                           const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
        bundle.set(kern_param.workspace_ptr);
        conv_func(bundle, kern_param, ncb_index);
    };
    ret_kerns.push_back({do_conv, ncb_range});

    return ret_kerns;
}
#endif