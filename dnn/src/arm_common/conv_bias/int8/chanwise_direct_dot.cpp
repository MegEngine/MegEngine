#include <arm_neon.h>
#include "src/arm_common/conv_bias/int8/algos.h"
#include "src/arm_common/conv_bias/int8/direct_kernels/dot_direct_nchw_large.h"
#include "src/common/unroll_macro.h"
#if MGB_ENABLE_DOT
#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_conv_bias_int8_direct_dot_large_kernel)

using namespace megdnn;
using namespace arm_common;

using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;
using NCBKernIndex = fallback::ConvBiasImpl::NCBKernIndex;

namespace {

class DirectConvRunner {
public:
    DirectConvRunner(size_t flt_size, size_t stride) {
        if (flt_size == 9 && stride == 1) {
            m_func = megdnn_dot_nchw_large_chanwise_direct_conv_9x9s1_oh4_ow16;
        } else if (flt_size == 9 && stride == 2) {
            m_func = megdnn_dot_nchw_large_chanwise_direct_conv_9x9s2_oh4_ow16;
        } else if (flt_size == 11 && stride == 1) {
            m_func = megdnn_dot_nchw_large_chanwise_direct_conv_11x11s1_oh4_ow16;
        } else {
            megdnn_assert(flt_size == 11 && stride == 2);
            m_func = megdnn_dot_nchw_large_chanwise_direct_conv_11x11s2_oh4_ow16;
        }
    }
    size_t get_round_fw(const ConvBiasImpl::NCBKernSizeParam& param) const {
        auto&& fm = param.filter_meta;
        auto FW = fm.spatial[1];
        return round_up((size_t)FW, m_block_k);
    }

    size_t get_round_iw(const ConvBiasImpl::NCBKernSizeParam& param) const {
        auto&& fm = param.filter_meta;
        size_t SW = fm.stride[1];
        size_t OW = param.osz[1];
        size_t round_ow = round_up(OW, m_block_ow);
        size_t round_fw = get_round_fw(param);
        size_t pad_iw = round_ow * SW - SW + round_fw;
        return round_up(pad_iw, m_align_iw);
    }

    size_t get_round_ih(const ConvBiasImpl::NCBKernSizeParam& param) const {
        auto&& fm = param.filter_meta;
        size_t SH = fm.stride[0];
        size_t OH = param.osz[0];
        auto FH = fm.spatial[0];
        size_t round_oh = round_up(OH, m_block_oh);
        return round_oh * SH - SH + FH;
    }

    WorkspaceBundle get_sub_bundle(const ConvBiasImpl::NCBKernSizeParam& param) const {
        auto&& fm = param.filter_meta;
        auto FH = fm.spatial[0];

        size_t round_filter = get_round_fw(param) * FH;
        size_t round_ih = get_round_ih(param);
        size_t round_iw = get_round_iw(param);
        size_t pad_src = round_iw * round_ih;
        return {nullptr, {pad_src, round_filter}};
    }

    WorkspaceBundle get_total_bundle(
            const ConvBiasImpl::NCBKernSizeParam& param) const {
        auto sub_bundle = get_sub_bundle(param);
        auto sub_bundle_size = sub_bundle.total_size_in_bytes();
        size_t nr_threads = param.nr_threads;
        SmallVector<size_t> sizes_in_bytes;
        for (size_t i = 0; i < nr_threads; ++i) {
            sizes_in_bytes.push_back(sub_bundle_size);
        }
        WorkspaceBundle total_bundle(nullptr, sizes_in_bytes);
        return total_bundle;
    }

    void run(
            const int8_t* pad_src_ptr, const int8_t* round_filter_ptr, int32_t bias,
            int8_t* dst_ptr, size_t OH, size_t OW, size_t pad_iw, float scale,
            int8_t relu_val) const {
        const size_t ow_end = OW / m_block_ow * m_block_ow;
        const size_t ow_remain = OW - ow_end;
        const size_t oh_end = OH / m_block_oh * m_block_oh;
        const size_t oh_remain = OH - oh_end;
        int8_t cache[4 * 16];
        for (size_t oh = 0; oh < oh_end; oh += m_block_oh) {
            for (size_t ow = 0; ow < ow_end; ow += m_block_ow) {
                m_func(pad_src_ptr, round_filter_ptr, bias, dst_ptr, oh, ow, OH, OW,
                       pad_iw, scale, relu_val);
            }
            if (ow_remain > 0) {
                m_func(pad_src_ptr, round_filter_ptr, bias,
                       &cache[0] - (oh * m_block_ow + ow_end), oh, ow_end, OH,
                       m_block_ow, pad_iw, scale, relu_val);
                for (size_t i = 0; i < m_block_oh; ++i) {
                    for (size_t j = 0; j < ow_remain; ++j) {
                        dst_ptr[(i + oh) * OW + (j + ow_end)] = cache[i * 16 + j];
                    }
                }
            }
        }
        if (oh_remain > 0) {
            for (size_t ow = 0; ow < ow_end; ow += m_block_ow) {
                m_func(pad_src_ptr, round_filter_ptr, bias,
                       &cache[0] - (oh_end * m_block_ow + ow), oh_end, ow, OH,
                       m_block_ow, pad_iw, scale, relu_val);
                for (size_t i = 0; i < oh_remain; ++i) {
                    for (size_t j = 0; j < m_block_ow; ++j) {
                        dst_ptr[(i + oh_end) * OW + (j + ow)] = cache[i * 16 + j];
                    }
                }
            }
            if (ow_remain > 0) {
                m_func(pad_src_ptr, round_filter_ptr, bias,
                       &cache[0] - (oh_end * m_block_ow + ow_end), oh_end, ow_end, OH,
                       m_block_ow, pad_iw, scale, relu_val);
                for (size_t i = 0; i < oh_remain; ++i) {
                    for (size_t j = 0; j < ow_remain; ++j) {
                        dst_ptr[(i + oh_end) * OW + (j + ow_end)] = cache[i * 16 + j];
                    }
                }
            }
        }
    }

private:
    std::function<void(
            const int8_t* src, const int8_t* weight, int32_t bias, int8_t* dst,
            size_t oh, size_t ow, size_t OH, size_t OW, size_t pad_iw,
            const float scale, int8_t relu_val)>
            m_func;
    size_t m_block_oh{4};
    size_t m_block_ow{16};
    size_t m_block_k{4};
    size_t m_align_iw{16};
};

void do_conv(
        const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
        const NCBKernIndex& ncb_index, const DirectConvRunner& runner) {
    auto&& fm = kern_param.filter_meta;
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t FH = fm.spatial[0];
    size_t FW = fm.spatial[1];

    float scale_bias = kern_param.bias_type.param<dtype::QuantizedS32>().scale;
    float scale_dst = kern_param.dst_type.param<dtype::QuantizedS8>().scale;
    float scale_dst_div = 1.f / scale_dst;
    size_t batch_id = ncb_index.ndrange_id[0];
    size_t group_id = ncb_index.ndrange_id[1];
    int8_t* pad_src_ptr = static_cast<int8_t*>(bundle.get(0));
    int8_t* round_filter_ptr = static_cast<int8_t*>(bundle.get(1));
    const int8_t* sptr = kern_param.src<dt_int8>(batch_id, group_id);
    const int32_t* bptr = kern_param.bias<dt_int32>(batch_id, group_id);
    const int8_t* fptr = kern_param.filter<dt_int8>(group_id);
    void* dst = kern_param.dst<void>(batch_id, group_id);
    size_t pad_iw = runner.get_round_iw(kern_param);

    memset(pad_src_ptr, 0, bundle.get_size(0));
    rep(ih, IH) {
        std::memcpy(
                pad_src_ptr + (ih + PH) * pad_iw + PW, sptr + ih * IW,
                sizeof(int8_t) * IW);
    }

    memset(round_filter_ptr, 0, bundle.get_size(1));
    size_t round_fw = runner.get_round_fw(kern_param);
    for (size_t fh = 0; fh < FH; ++fh) {
        std::memcpy(round_filter_ptr + fh * round_fw, fptr + fh * FW, FW);
    }

    int8_t relu_val = kern_param.nonlineMode == NonlineMode::RELU ? 0 : -128;
    int32_t bias_val = kern_param.bias_mode == BiasMode::NO_BIAS ? 0 : *bptr;

    int8_t* dst_ptr = (int8_t*)dst;
    runner.run(
            pad_src_ptr, round_filter_ptr, bias_val, dst_ptr, OH, OW, pad_iw,
            scale_bias * scale_dst_div, relu_val);
}

}  // namespace

bool ConvBiasImpl::AlgoDotS8DirectChanWiseLarge::usable(
        const NCBKernSizeParam& param, AlgoSelectionStrategy) const {
    if (!cpuinfo_has_arm_neon_dot()) {
        return false;
    }
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    auto FW = fm.spatial[1];
    auto SH = fm.stride[0];
    auto SW = fm.stride[1];
    auto noline = param.nonlineMode;
    auto bias_mode = param.bias_mode;
    bool avaible =
            //! src and filter are qint8, dst is qint8 or qint32
            (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
             param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
             (param.dst_type.enumv() == DTypeEnum::QuantizedS8)) &&
            fm.format == param::Convolution::Format::NCHW && !fm.should_flip &&
            (noline == NonlineMode::IDENTITY || noline == NonlineMode::RELU) &&
            (bias_mode == BiasMode::NO_BIAS ||
             bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) &&
            fm.spatial_ndim == 2 && fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
            SH == SW && (SH == 1 || SH == 2) && FH == FW && (FH == 9 || FH == 11) &&
            fm.icpg == 1 && fm.ocpg == 1;
    return avaible;
}

size_t ConvBiasImpl::AlgoDotS8DirectChanWiseLarge::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(
            megdnn_arm_common_conv_bias_int8_direct_dot_large_kernel,
            midout_iv("AlgoDotS8DirectChanWiseLarge::get_workspace"_hash)) {
        auto&& fm = param.filter_meta;
        DirectConvRunner runner(fm.spatial[0], fm.stride[0]);
        auto total_bundle = runner.get_total_bundle(param);
        return total_bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoDotS8DirectChanWiseLarge::
        dispatch_kerns(const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(
            megdnn_arm_common_conv_bias_int8_direct_dot_large_kernel,
            midout_iv("AlgoDotS8DirectChanWiseLarge::dispatch_kerns"_hash)) {
        SmallVector<ConvBiasImpl::NCBKern> ret_kerns;
        auto&& fm = param.filter_meta;
        DirectConvRunner runner(fm.spatial[0], fm.stride[0]);
        WorkspaceBundle wbundle = runner.get_sub_bundle(param);
        WorkspaceBundle total_bundle = runner.get_total_bundle(param);

        auto exec_one_group = [wbundle, total_bundle, runner](
                                      const NCBKernParam& kern_param,
                                      const NCBKernIndex& ncb_index) mutable {
            WorkspaceBundle temp_total_bundle = total_bundle;
            temp_total_bundle.set(kern_param.workspace_ptr);
            WorkspaceBundle temp_bundle = wbundle;
            temp_bundle.set(temp_total_bundle.get(ncb_index.thread_id));
            do_conv(temp_bundle, kern_param, ncb_index, runner);
        };
        size_t N = param.n;
        size_t group = fm.group;
        ret_kerns.push_back({exec_one_group, {N, group}});
        return ret_kerns;
    }
    MIDOUT_END();
    return {};
}

#endif
