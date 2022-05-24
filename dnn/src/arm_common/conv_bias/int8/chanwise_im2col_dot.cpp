#include "megdnn/arch.h"
#if MGB_ENABLE_DOT
#include "src/arm_common/simd_macro/marm_neon.h"

#include "src/arm_common/conv_bias/int8/algos.h"
#include "src/arm_common/matrix_mul/int8/gemv.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_conv_bias_int8_im2col_dot_large_kernel)

using namespace megdnn;
using namespace arm_common;

using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;
using NCBKernIndex = fallback::ConvBiasImpl::NCBKernIndex;

namespace {
constexpr size_t block_n = 32;
constexpr size_t block_k = 4;

WorkspaceBundle get_sub_bundle(const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto OH = param.osz[0];
    auto OW = param.osz[1];
    size_t IH = param.isz[0];
    size_t IW = param.isz[1];
    auto FH = fm.spatial[0];
    auto FW = fm.spatial[1];
    size_t PH = param.filter_meta.padding[0];
    size_t PW = param.filter_meta.padding[1];

    size_t round_ohw = round_up((size_t)OH * OW, block_n);
    size_t round_filter = round_up((size_t)FW, block_k) * FH;
    size_t pad_src = (IW + PW * 2) * (IH + PH * 2);
    return {nullptr,
            {pad_src, round_filter, round_ohw * round_filter,
             round_ohw * sizeof(int32_t)}};
}

WorkspaceBundle get_total_bundle(const ConvBiasImpl::NCBKernSizeParam& param) {
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

template <size_t flt_size, size_t stride>
void im2col(
        const int8_t* src, int8_t* dst, size_t OH, size_t OW, size_t pad_iw,
        size_t round_filter) {
    constexpr size_t FH = flt_size;
    constexpr size_t FW = flt_size;
    constexpr size_t SH = stride;
    constexpr size_t SW = stride;
    constexpr size_t FW_ROUND = (FW + 3) / 4 * 4;
    int bn = 0;
    int ni = 0;
    for (size_t oh = 0; oh < OH; ++oh)
        for (size_t ow = 0; ow < OW; ++ow) {
            const int8_t* src_n = src + oh * SH * pad_iw + ow * SW;
            int bk = 0;
            int ki = 0;
            for (size_t fh = 0; fh < FH; ++fh)
                for (size_t fw = 0; fw < FW_ROUND; ++fw) {
                    dst[bn * block_n * round_filter + bk * block_n * block_k +
                        ni * block_k + ki] = src_n[fh * pad_iw + fw];
                    ++ki;
                    if (ki == block_k) {
                        ki = 0;
                        ++bk;
                    }
                }
            ++ni;
            if (ni == block_n) {
                ni = 0;
                ++bn;
            }
        }
}

template <>
void im2col<9, 1>(
        const int8_t* src, int8_t* dst, size_t OH, size_t OW, size_t pad_iw,
        size_t round_filter) {
    constexpr size_t FH = 9;
    constexpr size_t SH = 1;
    constexpr size_t SW = 1;
    constexpr size_t k_block_stride = block_k * block_n;

    constexpr size_t ow_block = 16;
    static const uint8_t tbl_array_0[16] = {0, 1, 2, 3, 1, 2, 3, 4,
                                            2, 3, 4, 5, 3, 4, 5, 6};
    static const uint8_t tbl_array_1[16] = {4, 5, 6, 7, 5, 6, 7, 8,
                                            6, 7, 8, 9, 7, 8, 9, 10};
    static const uint8_t tbl_array_2[16] = {8,  9,  10, 11, 9,  10, 11, 12,
                                            10, 11, 12, 13, 11, 12, 13, 14};

    uint8x16_t tbl_reg_0 = vld1q_u8(&tbl_array_0[0]);
    uint8x16_t tbl_reg_1 = vld1q_u8(&tbl_array_1[0]);
    uint8x16_t tbl_reg_2 = vld1q_u8(&tbl_array_2[0]);

    int bn = 0;
    int ni = 0;
    for (size_t oh = 0; oh < OH; ++oh)

        for (size_t ow = 0; ow < OW;) {
            if (ow + ow_block <= OW && ni + ow_block <= block_n) {
                const int8_t* src_n = src + oh * SH * pad_iw + ow * SW;
                int8_t* dst_n = dst + bn * block_n * round_filter + ni * block_k;
                for (size_t fh = 0; fh < FH; ++fh) {
                    int8x16_t read_w[2];
                    read_w[0] = vld1q_s8(src_n);
                    read_w[1] = vld1q_s8(src_n + 16);

                    int8x16_t n0123_0 = vqtbl1q_s8(read_w[0], tbl_reg_0);
                    int8x16_t n4567_0 = vqtbl1q_s8(read_w[0], tbl_reg_1);
                    int8x16_t n89ab_0 = vqtbl1q_s8(read_w[0], tbl_reg_2);
                    int8x16_t ncdef_0 =
                            vqtbl1q_s8(vextq_s8(read_w[0], read_w[1], 12), tbl_reg_0);

                    int8x16_t n0123_1 = n4567_0;
                    int8x16_t n4567_1 = n89ab_0;
                    int8x16_t n89ab_1 = ncdef_0;
                    int8x16_t ncdef_1 = vqtbl1q_s8(read_w[1], tbl_reg_0);

                    int8x16_t n0123_2 = n89ab_0;
                    int8x16_t n4567_2 = ncdef_0;
                    int8x16_t n89ab_2 = ncdef_1;
                    int8x16_t ncdef_2 = vqtbl1q_s8(read_w[1], tbl_reg_1);

                    vst1q_s8(dst_n + 0 * 16, n0123_0);
                    vst1q_s8(dst_n + 1 * 16, n4567_0);
                    vst1q_s8(dst_n + 2 * 16, n89ab_0);
                    vst1q_s8(dst_n + 3 * 16, ncdef_0);

                    vst1q_s8(dst_n + 1 * k_block_stride + 0 * 16, n0123_1);
                    vst1q_s8(dst_n + 1 * k_block_stride + 1 * 16, n4567_1);
                    vst1q_s8(dst_n + 1 * k_block_stride + 2 * 16, n89ab_1);
                    vst1q_s8(dst_n + 1 * k_block_stride + 3 * 16, ncdef_1);

                    vst1q_s8(dst_n + 2 * k_block_stride + 0 * 16, n0123_2);
                    vst1q_s8(dst_n + 2 * k_block_stride + 1 * 16, n4567_2);
                    vst1q_s8(dst_n + 2 * k_block_stride + 2 * 16, n89ab_2);
                    vst1q_s8(dst_n + 2 * k_block_stride + 3 * 16, ncdef_2);

                    dst_n += 3 * k_block_stride;
                    src_n += pad_iw;
                }
                ni += ow_block;
                ow += ow_block;
                if (ni == block_n) {
                    ni = 0;
                    ++bn;
                }
            } else {
                const int8_t* src_n = src + oh * SH * pad_iw + ow * SW;
                int8_t* dst_n = dst + bn * block_n * round_filter + ni * block_k;
                for (size_t fh = 0; fh < FH; ++fh) {
                    int8x16_t read_w[0];
                    read_w[0] = vld1q_s8(src_n);
                    vst1q_lane_s32(dst_n, read_w[0], 0);
                    vst1q_lane_s32(dst_n + 1 * k_block_stride, read_w[0], 1);
                    vst1q_lane_s32(dst_n + 2 * k_block_stride, read_w[0], 2);
                    dst_n += 3 * k_block_stride;
                    src_n += pad_iw;
                }
                ++ni;
                ++ow;
                if (ni == block_n) {
                    ni = 0;
                    ++bn;
                }
            }
        }
}

template <>
void im2col<9, 2>(
        const int8_t* src, int8_t* dst, size_t OH, size_t OW, size_t pad_iw,
        size_t round_filter) {
    constexpr size_t FH = 9;
    constexpr size_t SH = 2;
    constexpr size_t SW = 2;
    constexpr size_t k_block_stride = block_k * block_n;

    constexpr size_t ow_block = 16;
    static const uint8_t tbl_array_0[16] = {0, 1, 2, 3, 2, 3, 4, 5,
                                            4, 5, 6, 7, 6, 7, 8, 9};
    static const uint8_t tbl_array_1[16] = {4, 5, 6,  7,  6,  7,  8,  9,
                                            8, 9, 10, 11, 10, 11, 12, 13};

    uint8x16_t tbl_reg_0 = vld1q_u8(&tbl_array_0[0]);
    uint8x16_t tbl_reg_1 = vld1q_u8(&tbl_array_1[0]);

    int bn = 0;
    int ni = 0;
    for (size_t oh = 0; oh < OH; ++oh)
        for (size_t ow = 0; ow < OW;) {
            if (ow + ow_block <= OW && ni + ow_block <= block_n) {
                const int8_t* src_n = src + oh * SH * pad_iw + ow * SW;
                int8_t* dst_n = dst + bn * block_n * round_filter + ni * block_k;
                for (size_t fh = 0; fh < FH; ++fh) {
                    int8x16_t read_w[3];
                    read_w[0] = vld1q_s8(src_n);
                    read_w[1] = vld1q_s8(src_n + 16);
                    read_w[2] = vld1q_s8(src_n + 32);
                    int8x16_t ext_8 = vextq_s8(read_w[0], read_w[1], 8);
                    int8x16_t ext_24 = vextq_s8(read_w[1], read_w[2], 8);

                    int8x16_t n0123_0 = vqtbl1q_s8(read_w[0], tbl_reg_0);
                    int8x16_t n4567_0 = vqtbl1q_s8(ext_8, tbl_reg_0);
                    int8x16_t n89ab_0 = vqtbl1q_s8(read_w[1], tbl_reg_0);
                    int8x16_t ncdef_0 = vqtbl1q_s8(ext_24, tbl_reg_0);

                    int8x16_t n0123_1 = vqtbl1q_s8(read_w[0], tbl_reg_1);
                    int8x16_t n4567_1 = vqtbl1q_s8(ext_8, tbl_reg_1);
                    int8x16_t n89ab_1 = vqtbl1q_s8(read_w[1], tbl_reg_1);
                    int8x16_t ncdef_1 = vqtbl1q_s8(ext_24, tbl_reg_1);

                    int8x16_t n0123_2 = n4567_0;
                    int8x16_t n4567_2 = n89ab_0;
                    int8x16_t n89ab_2 = ncdef_0;
                    int8x16_t ncdef_2 = vqtbl1q_s8(read_w[2], tbl_reg_0);

                    vst1q_s8(dst_n + 0 * 16, n0123_0);
                    vst1q_s8(dst_n + 1 * 16, n4567_0);
                    vst1q_s8(dst_n + 2 * 16, n89ab_0);
                    vst1q_s8(dst_n + 3 * 16, ncdef_0);

                    vst1q_s8(dst_n + 1 * k_block_stride + 0 * 16, n0123_1);
                    vst1q_s8(dst_n + 1 * k_block_stride + 1 * 16, n4567_1);
                    vst1q_s8(dst_n + 1 * k_block_stride + 2 * 16, n89ab_1);
                    vst1q_s8(dst_n + 1 * k_block_stride + 3 * 16, ncdef_1);

                    vst1q_s8(dst_n + 2 * k_block_stride + 0 * 16, n0123_2);
                    vst1q_s8(dst_n + 2 * k_block_stride + 1 * 16, n4567_2);
                    vst1q_s8(dst_n + 2 * k_block_stride + 2 * 16, n89ab_2);
                    vst1q_s8(dst_n + 2 * k_block_stride + 3 * 16, ncdef_2);

                    dst_n += 3 * k_block_stride;
                    src_n += pad_iw;
                }
                ni += ow_block;
                ow += ow_block;
                if (ni == block_n) {
                    ni = 0;
                    ++bn;
                }
            } else {
                const int8_t* src_n = src + oh * SH * pad_iw + ow * SW;
                int8_t* dst_n = dst + bn * block_n * round_filter + ni * block_k;
                for (size_t fh = 0; fh < FH; ++fh) {
                    int8x16_t read_w[0];
                    read_w[0] = vld1q_s8(src_n);
                    vst1q_lane_s32(dst_n, read_w[0], 0);
                    vst1q_lane_s32(dst_n + 1 * k_block_stride, read_w[0], 1);
                    vst1q_lane_s32(dst_n + 2 * k_block_stride, read_w[0], 2);
                    dst_n += 3 * k_block_stride;
                    src_n += pad_iw;
                }
                ++ni;
                ++ow;
                if (ni == block_n) {
                    ni = 0;
                    ++bn;
                }
            }
        }
}

void do_conv(
        const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
        const NCBKernIndex& ncb_index) {
    auto&& fm = kern_param.filter_meta;
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t FH = fm.spatial[0];
    size_t FW = fm.spatial[1];
    size_t SH = fm.stride[0];

    float scale_bias = kern_param.bias_type.param<dtype::QuantizedS32>().scale;
    float scale_dst = kern_param.dst_type.param<dtype::QuantizedS8>().scale;
    float scale_dst_div = 1.f / scale_dst;

    size_t batch_id = ncb_index.ndrange_id[0];
    size_t group_id = ncb_index.ndrange_id[1];
    int8_t* pad_src_ptr = static_cast<int8_t*>(bundle.get(0));
    int8_t* round_filter_ptr = static_cast<int8_t*>(bundle.get(1));
    int8_t* im2col_ptr = static_cast<int8_t*>(bundle.get(2));
    int32_t* i32_ptr = static_cast<int32_t*>(bundle.get(3));
    const int8_t* sptr = kern_param.src<dt_int8>(batch_id, group_id);
    const int32_t* bptr = kern_param.bias<dt_int32>(batch_id, group_id);
    const int8_t* fptr = kern_param.filter<dt_int8>(group_id);
    void* dst = kern_param.dst<void>(batch_id, group_id);

    size_t round_filter = round_up(FW, block_k) * FH;
    size_t pad_iw = IW + 2 * PW;

    memset(pad_src_ptr, 0, bundle.get_size(0));
    rep(ih, IH) {
        std::memcpy(
                pad_src_ptr + (ih + PH) * pad_iw + PW, sptr + ih * IW,
                sizeof(int8_t) * IW);
    }

    memset(round_filter_ptr, 0, bundle.get_size(1));
    size_t fh_stride = round_up(FW, block_k);
    for (size_t fh = 0; fh < FH; ++fh) {
        std::memcpy(round_filter_ptr + fh * fh_stride, fptr + fh * FW, FW);
    }

    memset(im2col_ptr, 0, bundle.get_size(2));
    if (SH == 1) {
        im2col<9, 1>(pad_src_ptr, im2col_ptr, OH, OW, pad_iw, round_filter);
    } else {
        im2col<9, 2>(pad_src_ptr, im2col_ptr, OH, OW, pad_iw, round_filter);
    }

    gevm_naive_n32k4_dot(
            round_filter_ptr, im2col_ptr, i32_ptr, 1, OH * OW, round_filter, 0, 0, 0);

    int32_t bias_val = kern_param.bias_mode == BiasMode::NO_BIAS ? 0 : *bptr;
    int8_t relu_val = kern_param.nonlineMode == NonlineMode::RELU ? 0 : -128;
    int8_t* dst_ptr = (int8_t*)dst;
    for (size_t i = 0; i < OH * OW; ++i) {
        //! optimize by tbl
        int val = roundf(scale_bias * scale_dst_div * (i32_ptr[i] + bias_val));
        val = val < -128 ? -128 : val;
        val = val > 127 ? 127 : val;
        val = val > relu_val ? val : relu_val;
        dst_ptr[i] = val;
    }
}

}  // namespace

bool ConvBiasImpl::AlgoDotS8Im2colChanWiseLarge::usable(
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
            SH == SW && (SH == 1 || SH == 2) && FH == FW && (FH == 9) && fm.icpg == 1 &&
            fm.ocpg == 1;
    return avaible;
}

size_t ConvBiasImpl::AlgoDotS8Im2colChanWiseLarge::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(
            megdnn_arm_common_conv_bias_int8_im2col_dot_large_kernel,
            midout_iv("AlgoDotS8Im2colChanWiseLarge::get_workspace"_hash)) {
        auto bundle = get_total_bundle(param);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoDotS8Im2colChanWiseLarge::
        dispatch_kerns(const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(
            megdnn_arm_common_conv_bias_int8_im2col_dot_large_kernel,
            midout_iv("AlgoDotS8Im2colChanWiseLarge::dispatch_kerns"_hash)) {
        SmallVector<ConvBiasImpl::NCBKern> ret_kerns;
        auto fm = param.filter_meta;
        size_t N = param.n;
        size_t group = fm.group;
        WorkspaceBundle wbundle = get_sub_bundle(param);
        WorkspaceBundle total_bundle = get_total_bundle(param);
        auto exec_one_group = [wbundle, total_bundle](
                                      const NCBKernParam& kern_param,
                                      const NCBKernIndex& ncb_index) mutable {
            WorkspaceBundle temp_total_bundle = total_bundle;
            temp_total_bundle.set(kern_param.workspace_ptr);
            WorkspaceBundle temp_bundle = wbundle;
            temp_bundle.set(temp_total_bundle.get(ncb_index.thread_id));
            do_conv(temp_bundle, kern_param, ncb_index);
        };
        ret_kerns.push_back({exec_one_group, {N, group}});
        return ret_kerns;
    }
    MIDOUT_END();
    return {};
}
#endif