/**
 * \file dnn/src/x86/conv_bias/int8/avx2_direct_conv_stride2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/x86/conv_bias/int8/avx2_direct_conv_stride2.h"
#include "src/x86/conv_bias/int8/common_helper.h"
#include "src/x86/conv_bias/postprocess_helper.h"

namespace megdnn {
namespace x86 {
namespace direct_conv_avx2_stride2 {

//! layout:(N,IC,IH,IW)-->(N,IC/2,H,2*W_envnW_odd)
MEGDNN_ATTRIBUTE_TARGET("sse4.1")
void pack_src_conv_avx2_stride2(const WorkspaceBundle& bundle,
                                const ConvBiasImpl::NCBKernParam& kern_param,
                                const ConvBiasImpl::NCBKernIndex& ncb_index) {
    int32_t ih = kern_param.isz[0];
    int32_t iw = kern_param.isz[1];
    int32_t ic = kern_param.filter_meta.icpg;
    int32_t pad_h = kern_param.filter_meta.padding[0];
    int32_t pad_w = kern_param.filter_meta.padding[1];

    constexpr int ic_step = 2;
    constexpr int iw_step = 16;
    const int ic_end = ic / ic_step * ic_step;
    const int iw_end = iw / iw_step * iw_step;
    const int iw_remain = iw - iw_end;
    const int out_h = ih + 2 * pad_h;
    const int out_w = (iw + pad_w * 2) * ic_step;
    const int c_stride = ih * iw;
    int8_t zero[iw_step]{0};
    size_t group = kern_param.filter_meta.group;
    size_t packed_group_size = out_h * out_w * div_ceil(ic, ic_step);

    size_t group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1],
           channel_id = ncb_index.ndrange_id[2];

    const int8_t* src_ptr = kern_param.src<int8_t>(batch_id, group_id) +
                            ic_step * channel_id * c_stride;
    int8_t* packed_src = static_cast<int8_t*>(bundle.get(0)) +
                         batch_id * group * packed_group_size +
                         group_id * packed_group_size +
                         channel_id * out_w * out_h;

    auto ic_count = ic_step * static_cast<int>(channel_id);
    // default pad len for pad even and odd
    auto pad_even_tail_len = pad_w / 2 * ic_step;
    auto pad_odd_tail_len = (pad_w + 1) / 2 * ic_step;
    auto pad_odd_head_len = pad_w / 2 * ic_step;
    auto pad_even_head_len = (pad_w + 1) / 2 * ic_step;
    if (ic_count < ic_end) {
        auto src_ptr_ic0 = src_ptr;
        auto src_ptr_ic1 = src_ptr_ic0 + c_stride;
        append_zero(packed_src, pad_h * out_w);
        packed_src += pad_h * out_w;
        for (int h_iter = 0; h_iter < ih; ++h_iter) {
            auto out_ptr_even = packed_src + h_iter * out_w;
            auto out_ptr_odd =
                    out_ptr_even + div_ceil(iw + 2 * pad_w, 2) * ic_step;
            append_zero_and_inc(out_ptr_even, pad_even_head_len);
            append_zero_and_inc(out_ptr_odd, pad_odd_head_len);
            for (int w_iter = 0; w_iter < iw_end; w_iter += iw_step) {
                if (pad_w % 2)
                    transpose_2x16_int8_odd_even(src_ptr_ic0, src_ptr_ic1,
                                                 out_ptr_odd, out_ptr_even);
                else
                    transpose_2x16_int8_odd_even(src_ptr_ic0, src_ptr_ic1,
                                                 out_ptr_even, out_ptr_odd);
                out_ptr_even += iw_step;
                out_ptr_odd += iw_step;
                src_ptr_ic0 += iw_step;
                src_ptr_ic1 += iw_step;
            }
            if (iw_remain > 0) {
                if (iw_remain % 2) {
                    pad_odd_tail_len = pad_w / 2 * ic_step;
                    pad_even_tail_len = (pad_w + 1) / 2 * ic_step;
                }
                auto tmp_e = round_up(iw_remain, ic_step);
                if (pad_w % 2) {
                    transpose_2xn_int8_odd_even(src_ptr_ic0, src_ptr_ic1,
                                                out_ptr_odd, out_ptr_even,
                                                iw_remain);
                    out_ptr_even += iw_remain * ic_step - tmp_e;
                    out_ptr_odd += tmp_e;
                } else {
                    transpose_2xn_int8_odd_even(src_ptr_ic0, src_ptr_ic1,
                                                out_ptr_even, out_ptr_odd,
                                                iw_remain);
                    out_ptr_odd += iw_remain * ic_step - tmp_e;
                    out_ptr_even += tmp_e;
                }
                src_ptr_ic0 += iw_remain;
                src_ptr_ic1 += iw_remain;
            }
            append_zero_and_inc(out_ptr_even, pad_even_tail_len);
            append_zero_and_inc(out_ptr_odd, pad_odd_tail_len);
        }
        packed_src += ih * out_w;
        append_zero_and_inc(packed_src, pad_h * out_w);
    } else {
        auto src_ptr_ic0 = src_ptr;
        auto src_ptr_ic1 = &zero[0];
        append_zero(packed_src, pad_h * out_w);
        packed_src += pad_h * out_w;
        for (int h_iter = 0; h_iter < ih; ++h_iter) {
            auto out_ptr_even = packed_src + h_iter * out_w;
            auto out_ptr_odd =
                    out_ptr_even + div_ceil(iw + 2 * pad_w, 2) * ic_step;
            append_zero_and_inc(out_ptr_even, pad_even_head_len);
            append_zero_and_inc(out_ptr_odd, pad_odd_head_len);
            for (int w_iter = 0; w_iter < iw_end; w_iter += iw_step) {
                if (pad_w % 2)
                    transpose_2x16_int8_odd_even(src_ptr_ic0, src_ptr_ic1,
                                                 out_ptr_odd, out_ptr_even);
                else
                    transpose_2x16_int8_odd_even(src_ptr_ic0, src_ptr_ic1,
                                                 out_ptr_even, out_ptr_odd);
                out_ptr_even += iw_step;
                out_ptr_odd += iw_step;
                src_ptr_ic0 += iw_step;
            }
            if (iw_remain > 0) {
                if (iw_remain % 2) {
                    pad_odd_tail_len = pad_w / 2 * ic_step;
                    pad_even_tail_len = (pad_w + 1) / 2 * ic_step;
                }
                auto tmp_e = round_up(iw_remain, ic_step);
                if (pad_w % 2) {
                    transpose_2xn_int8_odd_even(src_ptr_ic0, src_ptr_ic1,
                                                out_ptr_odd, out_ptr_even,
                                                iw_remain);
                    out_ptr_even += iw_remain * ic_step - tmp_e;
                    out_ptr_odd += tmp_e;
                } else {
                    transpose_2xn_int8_odd_even(src_ptr_ic0, src_ptr_ic1,
                                                out_ptr_even, out_ptr_odd,
                                                iw_remain);
                    out_ptr_odd += iw_remain * ic_step - tmp_e;
                    out_ptr_even += tmp_e;
                }
                src_ptr_ic0 += iw_remain;
            }
            append_zero_and_inc(out_ptr_even, pad_even_tail_len);
            append_zero_and_inc(out_ptr_odd, pad_odd_tail_len);
        }
        packed_src += ih * out_w;
        append_zero_and_inc(packed_src, pad_h * out_w);
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse4.1")
static inline void pack_filter_conv_avx2_stride2(
        const WorkspaceBundle& bundle, const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index) {
    MEGDNN_MARK_USED_VAR(ncb_index);
    int32_t oc = kern_param.filter_meta.ocpg;
    int32_t ic = kern_param.filter_meta.icpg;
    int32_t kh = kern_param.filter_meta.spatial[0];
    int32_t kw = kern_param.filter_meta.spatial[1];

    constexpr int k_step = 8;
    constexpr int ic_step = 2;
    constexpr int oc_step = 4;
    const int kernel_size = kh * kw;
    const int kernel_end = kernel_size / k_step * k_step;
    const int kernel_remain = kernel_size - kernel_end;
    const int ic_end = ic / ic_step * ic_step;
    const int ic_remain = ic - ic_end;
    const int oc_end = oc / oc_step * oc_step;
    const int oc_remain = oc - oc_end;
    const int oc_stride = ic * kh * kw;
    const int oc_out_stride = round_up(ic, ic_step) * kh * kw;
    const int8_t zero[k_step]{0};

    size_t group_id = ncb_index.ndrange_id[0],
           oc_index_id = ncb_index.ndrange_id[1];

    const int8_t* pack_filter_ptr = kern_param.filter<int8_t>(group_id);
    int16_t* out_ptr = static_cast<int16_t*>(bundle.get(1)) +
                       group_id * round_up(oc, oc_step) * oc_out_stride;

    auto pack_oc_step = [&]() {
        auto oc_out_ptr = out_ptr + oc_step * oc_index_id * oc_out_stride;
        for (int ic_iter = 0; ic_iter < ic_end; ic_iter += ic_step) {
            auto pack_filter_ptr_base = pack_filter_ptr +
                                        oc_step * oc_index_id * oc_stride +
                                        ic_iter * kernel_size;
            auto pack_filter_ptr_0_0 = pack_filter_ptr_base + 0 * oc_stride;
            auto pack_filter_ptr_0_1 = pack_filter_ptr_0_0 + kernel_size;
            auto pack_filter_ptr_1_0 = pack_filter_ptr_base + 1 * oc_stride;
            auto pack_filter_ptr_1_1 = pack_filter_ptr_1_0 + kernel_size;
            auto pack_filter_ptr_2_0 = pack_filter_ptr_base + 2 * oc_stride;
            auto pack_filter_ptr_2_1 = pack_filter_ptr_2_0 + kernel_size;
            auto pack_filter_ptr_3_0 = pack_filter_ptr_base + 3 * oc_stride;
            auto pack_filter_ptr_3_1 = pack_filter_ptr_3_0 + kernel_size;
            for (int k_iter = 0; k_iter < kernel_end; k_iter += k_step) {
                transpose_4x2x8_int8_int16(
                        pack_filter_ptr_0_0, pack_filter_ptr_0_1,
                        pack_filter_ptr_1_0, pack_filter_ptr_1_1,
                        pack_filter_ptr_2_0, pack_filter_ptr_2_1,
                        pack_filter_ptr_3_0, pack_filter_ptr_3_1, oc_out_ptr);
                oc_out_ptr += k_step * oc_step * ic_step;
                pack_filter_ptr_0_0 += k_step;
                pack_filter_ptr_0_1 += k_step;
                pack_filter_ptr_1_0 += k_step;
                pack_filter_ptr_1_1 += k_step;
                pack_filter_ptr_2_0 += k_step;
                pack_filter_ptr_2_1 += k_step;
                pack_filter_ptr_3_0 += k_step;
                pack_filter_ptr_3_1 += k_step;
            }
            if (kernel_remain > 0) {
                transpose_4x2xn_int8_int16(
                        pack_filter_ptr_0_0, pack_filter_ptr_0_1,
                        pack_filter_ptr_1_0, pack_filter_ptr_1_1,
                        pack_filter_ptr_2_0, pack_filter_ptr_2_1,
                        pack_filter_ptr_3_0, pack_filter_ptr_3_1, oc_out_ptr,
                        kernel_remain);
                oc_out_ptr += kernel_remain * oc_step * ic_step;
            }
        }
        if (ic_remain > 0) {
            auto pack_filter_ptr_base = pack_filter_ptr +
                                        oc_step * oc_index_id * oc_stride +
                                        ic_end * kernel_size;
            auto pack_filter_ptr_0_0 = pack_filter_ptr_base + 0 * oc_stride;
            auto pack_filter_ptr_0_1 = &zero[0];
            auto pack_filter_ptr_1_0 = pack_filter_ptr_base + 1 * oc_stride;
            auto pack_filter_ptr_1_1 = &zero[0];
            auto pack_filter_ptr_2_0 = pack_filter_ptr_base + 2 * oc_stride;
            auto pack_filter_ptr_2_1 = &zero[0];
            auto pack_filter_ptr_3_0 = pack_filter_ptr_base + 3 * oc_stride;
            auto pack_filter_ptr_3_1 = &zero[0];
            for (int k_iter = 0; k_iter < kernel_end; k_iter += k_step) {
                transpose_4x2x8_int8_int16(
                        pack_filter_ptr_0_0, pack_filter_ptr_0_1,
                        pack_filter_ptr_1_0, pack_filter_ptr_1_1,
                        pack_filter_ptr_2_0, pack_filter_ptr_2_1,
                        pack_filter_ptr_3_0, pack_filter_ptr_3_1, oc_out_ptr);
                oc_out_ptr += oc_step * k_step * 2;
                pack_filter_ptr_0_0 += k_step;
                pack_filter_ptr_1_0 += k_step;
                pack_filter_ptr_2_0 += k_step;
                pack_filter_ptr_3_0 += k_step;
            }
            if (kernel_remain > 0) {
                transpose_4x2xn_int8_int16(
                        pack_filter_ptr_0_0, pack_filter_ptr_0_1,
                        pack_filter_ptr_1_0, pack_filter_ptr_1_1,
                        pack_filter_ptr_2_0, pack_filter_ptr_2_1,
                        pack_filter_ptr_3_0, pack_filter_ptr_3_1, oc_out_ptr,
                        kernel_remain);
                oc_out_ptr += kernel_remain * 2;
            }
        }
    };
    auto pack_oc_remain = [&]() {
        auto oc_out_ptr = out_ptr + oc_end * oc_out_stride;
        for (int ic_iter = 0; ic_iter < ic_end; ic_iter += ic_step) {
            auto pack_filter_ptr_base = pack_filter_ptr + oc_end * oc_stride +
                                        ic_iter * kernel_size;
            auto pack_filter_ptr_0_0 = pack_filter_ptr_base + 0 * oc_stride;
            auto pack_filter_ptr_0_1 = pack_filter_ptr_0_0 + kernel_size;
            auto pack_filter_ptr_1_0 = &zero[0];
            auto pack_filter_ptr_1_1 = &zero[0];
            auto pack_filter_ptr_2_0 = &zero[0];
            auto pack_filter_ptr_2_1 = &zero[0];
            auto pack_filter_ptr_3_0 = &zero[0];
            auto pack_filter_ptr_3_1 = &zero[0];
            if (oc_remain >= 2) {
                pack_filter_ptr_1_0 = pack_filter_ptr_base + 1 * oc_stride;
                pack_filter_ptr_1_1 = pack_filter_ptr_1_0 + kernel_size;
            }
            if (oc_remain >= 3) {
                pack_filter_ptr_2_0 = pack_filter_ptr_base + 2 * oc_stride;
                pack_filter_ptr_2_1 = pack_filter_ptr_2_0 + kernel_size;
            }
            for (int k_iter = 0; k_iter < kernel_end; k_iter += k_step) {
                transpose_4x2x8_int8_int16(
                        pack_filter_ptr_0_0, pack_filter_ptr_0_1,
                        pack_filter_ptr_1_0, pack_filter_ptr_1_1,
                        pack_filter_ptr_2_0, pack_filter_ptr_2_1,
                        pack_filter_ptr_3_0, pack_filter_ptr_3_1, oc_out_ptr);
                oc_out_ptr += k_step * oc_step * ic_step;
                pack_filter_ptr_0_0 += k_step;
                pack_filter_ptr_0_1 += k_step;
                if (oc_remain >= 2) {
                    pack_filter_ptr_1_0 += k_step;
                    pack_filter_ptr_1_1 += k_step;
                }
                if (oc_remain >= 3) {
                    pack_filter_ptr_2_0 += k_step;
                    pack_filter_ptr_2_1 += k_step;
                }
            }
            if (kernel_remain > 0) {
                transpose_4x2xn_int8_int16(
                        pack_filter_ptr_0_0, pack_filter_ptr_0_1,
                        pack_filter_ptr_1_0, pack_filter_ptr_1_1,
                        pack_filter_ptr_2_0, pack_filter_ptr_2_1,
                        pack_filter_ptr_3_0, pack_filter_ptr_3_1, oc_out_ptr,
                        kernel_remain);
                oc_out_ptr += kernel_remain * oc_step * ic_step;
            }
        }
        if (ic_remain > 0) {
            auto pack_filter_ptr_base =
                    pack_filter_ptr + oc_end * oc_stride + ic_end * kernel_size;
            auto pack_filter_ptr_0_0 = pack_filter_ptr_base + 0 * oc_stride;
            auto pack_filter_ptr_0_1 = &zero[0];
            auto pack_filter_ptr_1_0 = &zero[0];
            auto pack_filter_ptr_1_1 = &zero[0];
            auto pack_filter_ptr_2_0 = &zero[0];
            auto pack_filter_ptr_2_1 = &zero[0];
            auto pack_filter_ptr_3_0 = &zero[0];
            auto pack_filter_ptr_3_1 = &zero[0];
            if (oc_remain >= 2) {
                pack_filter_ptr_1_0 = pack_filter_ptr_base + 1 * oc_stride;
            }
            if (oc_remain >= 3) {
                pack_filter_ptr_2_0 = pack_filter_ptr_base + 2 * oc_stride;
            }
            for (int k_iter = 0; k_iter < kernel_end; k_iter += k_step) {
                transpose_4x2x8_int8_int16(
                        pack_filter_ptr_0_0, pack_filter_ptr_0_1,
                        pack_filter_ptr_1_0, pack_filter_ptr_1_1,
                        pack_filter_ptr_2_0, pack_filter_ptr_2_1,
                        pack_filter_ptr_3_0, pack_filter_ptr_3_1, oc_out_ptr);
                oc_out_ptr += oc_step * k_step * 2;
                pack_filter_ptr_0_0 += k_step;
                if (oc_remain >= 2) {
                    pack_filter_ptr_1_0 += k_step;
                }
                if (oc_remain >= 3) {
                    pack_filter_ptr_2_0 += k_step;
                }
            }
            if (kernel_remain > 0) {
                transpose_4x2xn_int8_int16(
                        pack_filter_ptr_0_0, pack_filter_ptr_0_1,
                        pack_filter_ptr_1_0, pack_filter_ptr_1_1,
                        pack_filter_ptr_2_0, pack_filter_ptr_2_1,
                        pack_filter_ptr_3_0, pack_filter_ptr_3_1, oc_out_ptr,
                        kernel_remain);
                oc_out_ptr += kernel_remain * 2;
            }
        }
    };
    auto oc_count = oc_step * static_cast<int>(oc_index_id);
    if (oc_count < oc_end) {
        pack_oc_step();
    } else {
        pack_oc_remain();
    }
}

template <uint32_t oh_remain, uint32_t oc_remain, uint32_t ow_remain,
          uint32_t oc_step, uint32_t ic_step, uint32_t ow_step>
MEGDNN_ATTRIBUTE_TARGET("avx2")
static inline void kern_conv_avx2_stride2_normal_conv(
        const int16_t* pack_filter_ptr, const int8_t* pack_feat_ptr,
        const int ld_src, int32_t* c_ptr, const uint32_t ldoc, const int ic,
        const int ldic, const int ow, const uint32_t fw, const uint32_t fh) {
    megdnn_assert(oc_step == 4 && ic_step == 2 && ow_step == 8);
    __m256i filter_vec[2];
    __m256i feat_vec[2];
    __m256i c_temp[oc_step];
    __m256i c_vec[ow_step];
    c_vec[0] = _mm256_setzero_si256();
    c_vec[1] = _mm256_setzero_si256();
    c_vec[2] = _mm256_setzero_si256();
    c_vec[3] = _mm256_setzero_si256();
    c_vec[4] = _mm256_setzero_si256();
    c_vec[5] = _mm256_setzero_si256();
    c_vec[6] = _mm256_setzero_si256();
    c_vec[7] = _mm256_setzero_si256();
    constexpr unsigned int feat_offset_even_base = 2;

    auto feat_offset_odd_base = div_ceil(ld_src, 4) * 2;
    auto pack_feat_ptr_c = pack_feat_ptr;

    for (int iter_c = 0; iter_c < ic; iter_c += ic_step) {
        pack_feat_ptr = pack_feat_ptr_c + iter_c * ldic;
        for (uint32_t h_offset = 0; h_offset < fh; ++h_offset) {
            for (uint32_t w_offset = 0; w_offset < fw; ++w_offset) {
                auto feat_offset = feat_offset_even_base * (w_offset / 2) +
                                   feat_offset_odd_base * (w_offset % 2);
                feat_vec[0] = _mm256_cvtepi8_epi16_from_ptr(pack_feat_ptr +
                                                            feat_offset);
                if (!oh_remain) {
                    feat_vec[1] = _mm256_cvtepi8_epi16_from_ptr(
                            pack_feat_ptr + ld_src * 2 + feat_offset);
                }
                filter_vec[0] = _mm256_set1_epi32(*(int32_t*)(pack_filter_ptr));
                filter_vec[1] =
                        _mm256_set1_epi32(*(int32_t*)(pack_filter_ptr + 2));

#define CAL(o_i, f_i, s_i, interval)                                        \
    c_temp[o_i] = _mm256_madd_epi16(filter_vec[f_i], feat_vec[s_i]);        \
    c_vec[o_i + interval] =                                                 \
            _mm256_add_epi32(c_vec[o_i + interval], c_temp[o_i]);           \
    if ((0 == interval) || (0 == o_i) || (!oc_remain && (4 == interval))) { \
        if (!oh_remain) {                                                   \
            c_temp[o_i + 1] =                                               \
                    _mm256_madd_epi16(filter_vec[f_i], feat_vec[s_i + 1]);  \
            c_vec[o_i + 1 + interval] = _mm256_add_epi32(                   \
                    c_vec[o_i + 1 + interval], c_temp[o_i + 1]);            \
        }                                                                   \
    }

                CAL(0, 0, 0, 0);
                CAL(2, 1, 0, 0);
                filter_vec[0] =
                        _mm256_set1_epi32(*(int32_t*)(pack_filter_ptr + 4));
                if (!oc_remain) {
                    filter_vec[1] =
                            _mm256_set1_epi32(*(int32_t*)(pack_filter_ptr + 6));
                }
                CAL(0, 0, 0, 4);
                CAL(2, 1, 0, 4);
#undef CAL
                pack_filter_ptr += 8;
            }
            pack_feat_ptr += ld_src;
        }
    }
    if (ow_remain) {
        __m256i mask = _m256_continue_mask(ow_remain);
#define STORE(index)                                                        \
    if ((1 == index) || (oc_remain >= index || oc_remain == 0) ||           \
        (4 == index && !oc_remain)) {                                       \
        _mm256_maskstore_epi32((c_ptr + (index - 1) * ldoc), mask,          \
                               c_vec[(index - 1) * 2]);                     \
        if (!oh_remain) {                                                   \
            _mm256_maskstore_epi32((c_ptr + (index - 1) * ldoc + ow), mask, \
                                   c_vec[(index - 1) * 2 + 1]);             \
        }                                                                   \
    }
        STORE(1);
        STORE(2);
        STORE(3);
        STORE(4);
#undef STORE
    } else {
#define STORE(index)                                                         \
    if ((1 == index) || (oc_remain >= index || oc_remain == 0) ||            \
        (4 == index && !oc_remain)) {                                        \
        _mm256_storeu_si256((__m256i*)(c_ptr + (index - 1) * ldoc),          \
                            c_vec[(index - 1) * 2]);                         \
        if (!oh_remain) {                                                    \
            _mm256_storeu_si256((__m256i*)(c_ptr + (index - 1) * ldoc + ow), \
                                c_vec[(index - 1) * 2 + 1]);                 \
        }                                                                    \
    }
        STORE(1);
        STORE(2);
        STORE(3);
        STORE(4);
#undef STORE
    }
}
template <uint32_t oh_remain, uint32_t oc_remain, uint32_t ow_remain,
          uint32_t oc_step, uint32_t ic_step, uint32_t oh_step,
          uint32_t ow_step>
inline void block_kernel_entry(const int16_t* filter, const int8_t* src,
                               int32_t* dst, const uint32_t oc_end,
                               const uint32_t oc_index, const uint32_t oh_end,
                               const uint32_t ow_end,
                               const uint32_t pack_ic_stride,
                               const uint32_t pack_iw, const uint32_t oc_stride,
                               const ConvBiasImpl::NCBKernParam& kern_param) {
    auto fm = kern_param.filter_meta;
    const uint32_t ic = fm.icpg;
    const uint32_t fh = fm.spatial[0];
    const uint32_t fw = fm.spatial[1];
    const uint32_t ow = kern_param.osz[1];
    constexpr uint32_t stride_h = 2;

    if (oc_index < oc_end) {
        auto iter_dst_c_ptr = dst;
        auto iter_filter_ptr = filter;
        for (uint32_t oh_iter = 0; oh_iter < oh_end; oh_iter += oh_step) {
            for (uint32_t ow_iter = 0; ow_iter < ow_end; ow_iter += ow_step) {
                auto iter_dst_ptr = iter_dst_c_ptr + oh_iter * ow + ow_iter;
                auto iter_src_ptr =
                        src + oh_iter * stride_h * pack_iw + ow_iter * ic_step;
                kern_conv_avx2_stride2_normal_conv<0, 0, 0, oc_step, ic_step,
                                                   ow_step>(
                        iter_filter_ptr, iter_src_ptr, pack_iw, iter_dst_ptr,
                        oc_stride, ic, pack_ic_stride, ow, fw, fh);
            }
            if (ow_remain > 0) {
                auto iter_dst_ptr = iter_dst_c_ptr + oh_iter * ow + ow_end;
                auto iter_src_ptr =
                        src + oh_iter * stride_h * pack_iw + ow_end * ic_step;
                kern_conv_avx2_stride2_normal_conv<0, 0, ow_remain, oc_step,
                                                   ic_step, ow_step>(
                        iter_filter_ptr, iter_src_ptr, pack_iw, iter_dst_ptr,
                        oc_stride, ic, pack_ic_stride, ow, fw, fh);
            }
        }
        if (oh_remain > 0) {
            for (uint32_t ow_iter = 0; ow_iter < ow_end; ow_iter += ow_step) {
                auto iter_dst_ptr = iter_dst_c_ptr + oh_end * ow + ow_iter;
                auto iter_src_ptr =
                        src + oh_end * stride_h * pack_iw + ow_iter * ic_step;
                kern_conv_avx2_stride2_normal_conv<oh_remain, 0, 0, oc_step,
                                                   ic_step, ow_step>(
                        iter_filter_ptr, iter_src_ptr, pack_iw, iter_dst_ptr,
                        oc_stride, ic, pack_ic_stride, ow, fw, fh);
            }
            if (ow_remain > 0) {
                auto iter_dst_ptr = iter_dst_c_ptr + oh_end * ow + ow_end;
                auto iter_src_ptr =
                        src + oh_end * stride_h * pack_iw + ow_end * ic_step;
                kern_conv_avx2_stride2_normal_conv<oh_remain, 0, ow_remain,
                                                   oc_step, ic_step, ow_step>(
                        iter_filter_ptr, iter_src_ptr, pack_iw, iter_dst_ptr,
                        oc_stride, ic, pack_ic_stride, ow, fw, fh);
            }
        }
    } else {
        auto iter_dst_c_ptr = dst;
        auto iter_filter_ptr = filter;
        for (uint32_t oh_iter = 0; oh_iter < oh_end; oh_iter += oh_step) {
            for (uint32_t ow_iter = 0; ow_iter < ow_end; ow_iter += ow_step) {
                auto iter_dst_ptr = iter_dst_c_ptr + oh_iter * ow + ow_iter;
                auto iter_src_ptr =
                        src + oh_iter * stride_h * pack_iw + ow_iter * ic_step;
                kern_conv_avx2_stride2_normal_conv<0, oc_remain, 0, oc_step,
                                                   ic_step, ow_step>(
                        iter_filter_ptr, iter_src_ptr, pack_iw, iter_dst_ptr,
                        oc_stride, ic, pack_ic_stride, ow, fw, fh);
            }
            if (ow_remain > 0) {
                auto iter_dst_ptr = iter_dst_c_ptr + oh_iter * ow + ow_end;
                auto iter_src_ptr =
                        src + oh_iter * stride_h * pack_iw + ow_end * ic_step;
                kern_conv_avx2_stride2_normal_conv<0, oc_remain, ow_remain,
                                                   oc_step, ic_step, ow_step>(
                        iter_filter_ptr, iter_src_ptr, pack_iw, iter_dst_ptr,
                        oc_stride, ic, pack_ic_stride, ow, fw, fh);
            }
        }
        if (oh_remain > 0) {
            for (uint32_t ow_iter = 0; ow_iter < ow_end; ow_iter += ow_step) {
                auto iter_dst_ptr = iter_dst_c_ptr + oh_end * ow + ow_iter;
                auto iter_src_ptr =
                        src + oh_end * stride_h * pack_iw + ow_iter * ic_step;
                kern_conv_avx2_stride2_normal_conv<oh_remain, oc_remain, 0,
                                                   oc_step, ic_step, ow_step>(
                        iter_filter_ptr, iter_src_ptr, pack_iw, iter_dst_ptr,
                        oc_stride, ic, pack_ic_stride, ow, fw, fh);
            }
            if (ow_remain > 0) {
                auto iter_dst_ptr = iter_dst_c_ptr + oh_end * ow + ow_end;
                auto iter_src_ptr =
                        src + oh_end * stride_h * pack_iw + ow_end * ic_step;
                kern_conv_avx2_stride2_normal_conv<oh_remain, oc_remain,
                                                   ow_remain, oc_step, ic_step,
                                                   ow_step>(
                        iter_filter_ptr, iter_src_ptr, pack_iw, iter_dst_ptr,
                        oc_stride, ic, pack_ic_stride, ow, fw, fh);
            }
        }
    }
}

template <uint32_t oh_remain, uint32_t oc_remain, uint32_t oc_step,
          uint32_t ic_step, uint32_t oh_step, uint32_t ow_step>
inline void kernel_handle_ow_remain(
        uint32_t ow_remain, const int16_t* filter_ptr, const int8_t* feat_ptr,
        int32_t* dst_ptr, const uint32_t oc_end, const uint32_t oc_index,
        const uint32_t oh_end, const uint32_t ow_end,
        const uint32_t pack_ic_stride, const uint32_t pack_iw,
        const uint32_t oc_stride,
        const ConvBiasImpl::NCBKernParam& kern_param) {
#define cb(OW_REMAIN)                                                        \
    block_kernel_entry<oh_remain, oc_remain, OW_REMAIN, oc_step, ic_step,    \
                       oh_step, ow_step>(                                    \
            filter_ptr, feat_ptr, dst_ptr, oc_end, oc_index, oh_end, ow_end, \
            pack_ic_stride, pack_iw, oc_stride, kern_param);

#define cb_switch(_remain) \
    case _remain:          \
        cb(_remain);       \
        break;
    switch (ow_remain) {
        cb_switch(0);
        cb_switch(1);
        cb_switch(2);
        cb_switch(3);
        cb_switch(4);
        cb_switch(5);
        cb_switch(6);
        cb_switch(7);
        default:
            megdnn_assert(ow_remain <= 7);
            break;
    }
#undef cb_switch
#undef cb
}

template <uint32_t oh_remain, uint32_t oc_step, uint32_t ic_step,
          uint32_t oh_step, uint32_t ow_step>
inline void kernel_handle_oc_remain(
        uint32_t oc_remain, uint32_t ow_remain, const int16_t* filter_ptr,
        const int8_t* feat_ptr, int32_t* dst_ptr, const uint32_t oc_end,
        const uint32_t oc_index, const uint32_t oh_end, const uint32_t ow_end,
        const uint32_t pack_ic_stride, const uint32_t pack_iw,
        const uint32_t oc_stride,
        const ConvBiasImpl::NCBKernParam& kern_param) {
#define cb(OC_REMAIN)                                                        \
    kernel_handle_ow_remain<oh_remain, OC_REMAIN, oc_step, ic_step, oh_step, \
                            ow_step>(                                        \
            ow_remain, filter_ptr, feat_ptr, dst_ptr, oc_end, oc_index,      \
            oh_end, ow_end, pack_ic_stride, pack_iw, oc_stride, kern_param);

#define cb_switch(_remain) \
    case _remain:          \
        cb(_remain);       \
        break;
    switch (oc_remain) {
        cb_switch(0);
        cb_switch(1);
        cb_switch(2);
        cb_switch(3);
        default:
            megdnn_assert(oc_remain <= 3);
            break;
    }
#undef cb_switch
#undef cb
}

template <uint32_t oc_step, uint32_t ic_step, uint32_t oh_step,
          uint32_t ow_step>
inline void kernel_handle_oh_remain(
        uint32_t oh_remain, uint32_t oc_remain, uint32_t ow_remain,
        const int16_t* filter_ptr, const int8_t* feat_ptr, int32_t* dst_ptr,
        const uint32_t oc_end, const uint32_t oc_index, const uint32_t oh_end,
        const uint32_t ow_end, const uint32_t pack_ic_stride,
        const uint32_t pack_iw, const uint32_t oc_stride,
        const ConvBiasImpl::NCBKernParam& kern_param) {
#define cb(OH_REMAIN)                                                       \
    kernel_handle_oc_remain<OH_REMAIN, oc_step, ic_step, oh_step, ow_step>( \
            oc_remain, ow_remain, filter_ptr, feat_ptr, dst_ptr, oc_end,    \
            oc_index, oh_end, ow_end, pack_ic_stride, pack_iw, oc_stride,   \
            kern_param);

#define cb_switch(_remain) \
    case _remain:          \
        cb(_remain);       \
        break;
    switch (oh_remain) {
        cb_switch(0);
        cb_switch(1);
        default:
            megdnn_assert(oh_remain <= 1);
            break;
    }
#undef cb_switch
#undef cb
}
void kernel_imp(const WorkspaceBundle& bundle,
                const ConvBiasImpl::NCBKernParam& kern_param,
                const ConvBiasImpl::NCBKernIndex& ncb_index) {
    auto&& fm = kern_param.filter_meta;
    size_t group = fm.group;
    const uint32_t oc = fm.ocpg;
    const uint32_t oh = kern_param.osz[0];
    const uint32_t ow = kern_param.osz[1];
    const uint32_t ic = fm.icpg;
    const uint32_t ih = kern_param.isz[0];
    const uint32_t iw = kern_param.isz[1];
    const uint32_t kh = fm.spatial[0];
    const uint32_t kw = fm.spatial[1];
    const uint32_t pad_h = fm.padding[0];
    const uint32_t pad_w = fm.padding[1];

    constexpr uint32_t oc_step = 4;
    constexpr uint32_t ic_step = 2;
    constexpr uint32_t oh_step = 2;
    constexpr uint32_t ow_step = 8;

    const uint32_t filter_round_size = kh * kw * round_up(ic, ic_step);
    const uint32_t oc_stride = oh * ow;
    const uint32_t pack_iw = (iw + 2 * pad_w) * ic_step;
    const uint32_t pack_ih = ih + 2 * pad_h;
    const uint32_t pack_ic_stride = pack_iw * pack_ih / ic_step;
    const uint32_t packed_group_size =
            div_ceil(ic, ic_step) * pack_ih * pack_iw;

    size_t group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1],
           channel_id = ncb_index.ndrange_id[2];

    int8_t* src_ptr = static_cast<int8_t*>(bundle.get(0)) +
                      group_id * packed_group_size +
                      batch_id * group * packed_group_size;
    int16_t* filter_ptr = static_cast<int16_t*>(bundle.get(1)) +
                          group_id * round_up(oc, oc_step) * filter_round_size +
                          oc_step * channel_id * filter_round_size;

    bool need_post_process =
            kern_param.dst_type.enumv() == DTypeEnum::QuantizedS8;

    int32_t* dst_tptr = nullptr;
    if (need_post_process) {
        dst_tptr = static_cast<int32_t*>(bundle.get(2)) +
                   batch_id * group * oc * oc_stride +
                   group_id * oc * oc_stride + oc_step * channel_id * oh * ow;
    } else {
        dst_tptr = kern_param.dst<int32_t>(batch_id, group_id) +
                   oc_step * channel_id * oh * ow;
    }
    const uint32_t oc_end = oc / oc_step * oc_step;
    const uint32_t oc_remain = oc - oc_end;
    const uint32_t oh_end = oh / oh_step * oh_step;
    const uint32_t oh_remain = oh - oh_end;
    const uint32_t ow_end = ow / ow_step * ow_step;
    const uint32_t ow_remain = ow - ow_end;
    const uint32_t oc_index = oc_step * channel_id;

    kernel_handle_oh_remain<oc_step, ic_step, oh_step, ow_step>(
            oh_remain, oc_remain, ow_remain, filter_ptr, src_ptr, dst_tptr,
            oc_end, oc_index, oh_end, ow_end, pack_ic_stride, pack_iw,
            oc_stride, kern_param);
}

void do_post_process(const WorkspaceBundle& bundle,
                     const ConvBiasImpl::NCBKernParam& kern_param,
                     const ConvBiasImpl::NCBKernIndex& ncb_index) {
    auto&& fm = kern_param.filter_meta;
    const uint32_t group = fm.group;
    const uint32_t oc = fm.ocpg;
    const uint32_t oh = kern_param.osz[0];
    const uint32_t ow = kern_param.osz[1];

    size_t group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1];

    bool need_post_process =
            kern_param.dst_type.enumv() == DTypeEnum::QuantizedS8;
    void* dst_tptr = nullptr;
    if (need_post_process) {
        dst_tptr = static_cast<int32_t*>(bundle.get(2)) +
                   batch_id * group * oc * oh * ow +
                   group_id * oc * oh * ow;
    } else {
        dst_tptr = kern_param.dst<dt_int32>(batch_id, group_id);
    }
    void* dst_ptr = kern_param.dst<void>(batch_id, group_id);

#define cb(_bias_ctype, _dst_ctype, _postprocess_mode)                       \
    {                                                                        \
        const dt_int32* bias_ptr =                                           \
                kern_param.bias<dt_int32>(batch_id, group_id);               \
        PostProcess<DTypeTrait<_bias_ctype>::ctype,                          \
                    DTypeTrait<_dst_ctype>::ctype,                           \
                    _postprocess_mode>::run(dst_tptr,                        \
                                            const_cast<dt_int32*>(bias_ptr), \
                                            dst_ptr, kern_param.bias_mode,   \
                                            kern_param.nonlineMode,          \
                                            kern_param.bias_type,            \
                                            kern_param.dst_type, 1, oc, oh,  \
                                            ow);                             \
    }
    if (kern_param.src_type.enumv() == DTypeEnum::Int8 &&
        kern_param.filter_type.enumv() == DTypeEnum::Int8 &&
        kern_param.dst_type.enumv() == DTypeEnum::Int32) {
        cb(dt_int32, dt_int32, PostprocessMode::NO_PROCESS);
    } else if (kern_param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
               kern_param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
               kern_param.dst_type.enumv() == DTypeEnum::QuantizedS32) {
        cb(dtype::QuantizedS32, dtype::QuantizedS32,
           PostprocessMode::NO_PROCESS);
    } else if (kern_param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
               kern_param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
               kern_param.dst_type.enumv() == DTypeEnum::QuantizedS8) {
        cb(dtype::QuantizedS32, dtype::QuantizedS8, PostprocessMode::QUANTIZED);
    } else {
        megdnn_throw("unsupported data type on x86 avx2 direct conv algo");
    }
#undef cb
}

SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& kern_param,
                                const WorkspaceBundle& bundle) {
    SmallVector<NCBKern> ncb_kerns;
    auto fm = kern_param.filter_meta;
    size_t N = kern_param.n;
    size_t IC = kern_param.filter_meta.icpg;
    size_t OC = kern_param.filter_meta.ocpg;
    size_t group = fm.group;
#define cb(task)                                                               \
    auto task = [bundle = bundle, tmp_func](                                   \
                        const ConvBiasImpl::NCBKernParam& kern_param,          \
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable { \
        bundle.set(kern_param.workspace_ptr);                                  \
        tmp_func(bundle, kern_param,                                           \
                 {ncb_index.thread_id,                                         \
                  {ncb_index.ndrange_id[0], ncb_index.ndrange_id[1],           \
                   ncb_index.ndrange_id[2]}});                                 \
    };
    auto tmp_func = pack_src_conv_avx2_stride2;
    cb(pack_src_task);
    ncb_kerns.push_back({pack_src_task, {group, N, div_ceil(IC, 2_z)}});

    tmp_func = pack_filter_conv_avx2_stride2;
    cb(pack_filter_task);
    ncb_kerns.push_back({pack_filter_task, {group, div_ceil(OC, 4_z), 1_z}});

    tmp_func = kernel_imp;
    cb(conv_task);
    ncb_kerns.push_back({conv_task, {group, N, div_ceil(OC, 4_z)}});

    tmp_func = do_post_process;
    cb(post_process_task);
    ncb_kerns.push_back({post_process_task, {group, N, 1_z}});
#undef cb

    return ncb_kerns;
}

}  // namespace direct_conv_avx2_stride2
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
