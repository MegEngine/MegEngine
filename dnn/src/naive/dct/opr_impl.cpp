/**
 * \file dnn/src/naive/dct/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/naive/dct/opr_impl.h"
#include <cmath>
#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"
#include "src/naive/handle.h"
#include "src/naive/matrix_mul/matrix_mul_helper.h"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_dct_fwd)
namespace megdnn {
namespace naive {

namespace {

static inline void generate_c_matrix(float* result, int block) {
    constexpr float pi = M_PI;
    for (int i = 0; i < block; ++i) {
        for (int j = 0; j < block; ++j) {
            float alpha = i == 0 ? sqrt(1.f / static_cast<float>(block))
                                 : sqrt(2.f / static_cast<float>(block));
            result[i * block + j] = alpha * cos((2.f * j + 1.f) * i * pi /
                                                static_cast<float>(2 * block));
        }
    }
}

template <typename T>
void matmul(int m, int n, int k, int lda, int ldb, int ldc, const float* a,
            const T* b, float* c, bool trans_a, bool trans_b) {
    for (int m_idx = 0; m_idx < m; ++m_idx) {
        for (int n_idx = 0; n_idx < n; ++n_idx) {
            float res = 0.f;
            for (int k_idx = 0; k_idx < k; ++k_idx) {
                float av = trans_a ? a[k_idx * lda + m_idx]
                                   : a[m_idx * lda + k_idx];
                float bv = trans_b ? b[n_idx * ldb + k_idx]
                                   : b[k_idx * ldb + n_idx];
                res += av * bv;
            }
            c[m_idx * ldc + n_idx] = res;
        }
    }
}

std::vector<std::vector<int>> mask_offset_to_2dmask(
        _megdnn_tensor_in mask_offset, _megdnn_tensor_in mask_val) {
    std::vector<std::vector<int>> mask;
    if (mask_offset.layout.ndim > 0 && mask_offset.layout[0] >= 2) {
        const int offset_len = mask_offset.layout.shape[0];
        const int32_t* mask_offset_ptr = mask_offset.ptr<int32_t>();
        const int32_t* mask_val_ptr = mask_val.ptr<int32_t>();
        megdnn_assert(
                mask_val.layout.shape[0] ==
                        static_cast<size_t>(mask_offset_ptr[offset_len - 1]),
                "check mask offset %zu != %zu", mask_val.layout.shape[0],
                static_cast<size_t>(mask_offset_ptr[offset_len - 1]));

        for (int offset_idx = 1; offset_idx < offset_len; ++offset_idx) {
            mask.push_back({});
            const int mask_len = mask_offset_ptr[offset_idx] -
                                 mask_offset_ptr[offset_idx - 1];
            const int32_t* mask_ptr =
                    &mask_val_ptr[mask_offset_ptr[offset_idx - 1]];
            for (int val_idx = 0; val_idx < mask_len; ++val_idx) {
                mask[offset_idx - 1].push_back(mask_ptr[val_idx]);
            }
        }
    }
    return mask;
}

inline bool is_layout_nchw4(const TensorLayout& layout) {
    if (layout.ndim == 5 && layout[4] == 4) {
        return true;
    } else {
        return false;
    }
}

template <typename T>
using QuantizedCType =
        std::enable_if_t<DTypeTrait<T>::category == DTypeCategory::QUANTIZED,
                         typename DTypeTrait<T>::ctype>;

inline int8_t quant_float_2_int8(float val, DType dtype) {
    return dtype.param<::megdnn::dtype::QuantizedS8>().quantize(val).as_int8();
}

template <param::DctChannelSelect::Format format, typename Dtype>
inline void dct_output(Dtype* dst_ptr, const int oc_idx, const int img_size,
                       float val, DType) {
    dst_ptr[oc_idx * img_size] = val;
}
template <>
inline void dct_output<param::DctChannelSelect::Format::NCHW4>(
        int8_t* dst_ptr, const int oc_idx, const int img_size, float val,
        DType dtype) {
    dst_ptr[oc_idx / 4 * 4 * img_size + oc_idx % 4] =
            quant_float_2_int8(val, dtype);
}
template <param::DctChannelSelect::Format format>
struct ChannleBlock {
    static constexpr int block = 1;
};

template <>
struct ChannleBlock<param::DctChannelSelect::Format::NCHW4> {
    static constexpr int block = 4;
};

template <param::DctChannelSelect::Format format, typename Dtype>
void naive_dct(const uint8_t* src, Dtype* dst, int n, int c, int h, int w,
               int block, const std::vector<std::vector<int>>& mask,
               DType dtype) {
    constexpr int block_channel = ChannleBlock<format>::block;
    const int block_h = block;
    const int block_w = block;
    std::vector<float> c_matrix(block * block);
    std::vector<float> tmp(block * block);
    std::vector<float> tmp_result(block * block);
    generate_c_matrix(&c_matrix[0], block);
    megdnn_assert(h % block_h == 0, "h mod block_h == 0");
    megdnn_assert(w % block_w == 0, "w mod block_w == 0");
    const int oh = h / block_h;
    const int ow = w / block_w;
    const int o_img_size = oh * ow;
    std::vector<int> mask_offset;
    int mask_len_sum = 0;
    if (mask.size() > 0) {
        for (auto& sub_mask : mask) {
            mask_offset.push_back(mask_len_sum);
            mask_len_sum += sub_mask.size();
        }
    } else {
        for (int c_idx = 0; c_idx < c; ++c_idx) {
            mask_offset.push_back(mask_len_sum);
            mask_len_sum += block_h * block_w;
        }
    }
    const size_t o_batch_stride = mask_len_sum * oh * ow;

    for (int n_idx = 0; n_idx < n; ++n_idx) {
        for (int c_idx = 0; c_idx < c; ++c_idx) {
            megdnn_assert(mask_offset[c_idx] % block_channel == 0,
                          "%d mod %d == 0", mask_offset[c_idx], block_channel);
            const size_t src_offset = n_idx * c * h * w + c_idx * h * w;
            const uint8_t* src_channel = src + src_offset;
            const size_t dst_offset = n_idx * o_batch_stride +
                                      mask_offset[c_idx] / block_channel * oh *
                                              ow * block_channel;
            Dtype* dst_channel = dst + dst_offset;
            for (int oh_idx = 0; oh_idx < oh; ++oh_idx) {
                for (int ow_idx = 0; ow_idx < ow; ++ow_idx) {
                    matmul(block, block, block, block, w, block, &c_matrix[0],
                           &src_channel[oh_idx * block_h * w +
                                        ow_idx * block_w],
                           &tmp[0], false, false);
                    matmul(block, block, block, block, block, block, &tmp[0],
                           &c_matrix[0], &tmp_result[0], false, true);
                    Dtype* dst_start = dst_channel +
                                       (oh_idx * ow + ow_idx) * block_channel;
                    if (mask.size() == 0) {
                        for (int inner_h_idx = 0; inner_h_idx < block_h;
                             ++inner_h_idx) {
                            for (int inner_w_idx = 0; inner_w_idx < block_w;
                                 ++inner_w_idx) {
                                const int oc_idx =
                                        inner_h_idx * block_w + inner_w_idx;
                                dct_output<format>(
                                        dst_start, oc_idx, o_img_size,
                                        tmp_result[inner_h_idx * block +
                                                   inner_w_idx],
                                        dtype);
                            }
                        }
                    } else {
                        //! with mask
                        auto& sub_mask = mask[c_idx];
                        int dst_offset = 0;
                        for (auto mask_idx : sub_mask) {
                            dct_output<format>(dst_start, dst_offset,
                                               o_img_size, tmp_result[mask_idx],
                                               dtype);
                            ++dst_offset;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

void DctChannelSelectForwardImpl::exec(_megdnn_tensor_in src,
                                       _megdnn_tensor_in mask_offset,
                                       _megdnn_tensor_in mask_val,
                                       _megdnn_tensor_out dst,
                                       _megdnn_workspace /*workspace*/) {
    MIDOUT_BEGIN(megdnn_naive_dct_fwd) {
        int in = src.layout.shape[0];
        int ic = src.layout.shape[1];
        int ih = src.layout.shape[2];
        int iw = src.layout.shape[3];
        megdnn_assert(dst.raw_ptr, "dst can not be nullptr");
        const int block = param().dct_block_size;
        auto mask = mask_offset_to_2dmask(mask_offset, mask_val);
        if (dst.layout.dtype.enumv() == DTypeEnum::Float32) {
            megdnn_assert(!is_layout_nchw4(dst.layout) &&
                                  param().format == Param::Format::NCHW,
                          "dst must be nchw");
            MEGDNN_DISPATCH_CPU_KERN_OPR(naive_dct<Param::Format::NCHW>(
                    src.ptr<uint8_t>(), dst.ptr<float>(), in, ic, ih, iw, block,
                    mask, dst.layout.dtype));
        } else {
            megdnn_assert(dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8,
                          "dst must be q8");
            megdnn_assert(is_layout_nchw4(dst.layout) &&
                                  param().format == Param::Format::NCHW4,
                          "dst must be nchw4");
            MEGDNN_DISPATCH_CPU_KERN_OPR(naive_dct<Param::Format::NCHW4>(
                    src.ptr<uint8_t>(), static_cast<int8_t*>(dst.raw_ptr), in,
                    ic, ih, iw, block, mask, dst.layout.dtype));
        }
    }
    MIDOUT_END();
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
