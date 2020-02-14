/**
 * \file dnn/src/naive/relayout_format/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/relayout_format/opr_impl.h"
#include "src/naive/handle.h"

#include "megdnn/tensor_iter.h"

using namespace megdnn;
using namespace naive;

namespace {
template <typename dtype>
void padding_src_to_workspace(dtype* dptr, const dtype* sptr, size_t N,
                              size_t IC, size_t IH, size_t IW) {
    size_t IC4 = (IC + 3) / 4 * 4;
    size_t HW = IH * IW;
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < IC4; c++) {
            for (size_t idx = 0; idx < HW; idx++) {
                if (c < IC) {
                    *dptr = sptr[n * IC * HW + c * HW + idx];
                } else {
                    *dptr = 0;
                }
                dptr++;
            }
        }
    }
}

template <typename dtype>
void padding_to_workspace(dtype* dptr, const dtype* sptr,
                          const TensorLayout& src_layout, const size_t pad_axis,
                          const size_t align_size) {
    megdnn_assert(pad_axis < src_layout.ndim);
    const size_t axis_dim = src_layout[pad_axis];
    const size_t axis_dim_padded = round_up(axis_dim, align_size);
    const size_t axis_stride = src_layout.stride[pad_axis];
    const size_t repeat_number =
            src_layout.total_nr_elems() / (axis_dim * axis_stride);
    for (size_t outer_idx = 0; outer_idx < repeat_number; ++outer_idx) {
        const size_t dst_outer_offset =
                outer_idx * axis_dim_padded * axis_stride;
        const size_t src_inner_offset = outer_idx * axis_dim * axis_stride;
        for (size_t axis_idx = 0; axis_idx < axis_dim_padded; ++axis_idx) {
            for (size_t inner_idx = 0; inner_idx < axis_stride; ++inner_idx) {
                const size_t inner_idx_offset =
                        axis_idx * axis_stride + inner_idx;
                if (axis_idx < axis_dim) {
                    dptr[dst_outer_offset + inner_idx_offset] =
                            sptr[src_inner_offset + inner_idx_offset];
                } else {
                    dptr[dst_outer_offset + inner_idx_offset] =
                            static_cast<dtype>(0);
                }
            }
        }
    }
}
void padding_to_workspace(_megdnn_tensor_out dst, _megdnn_tensor_in src,
                          const size_t pad_axis, const size_t align_size) {
    switch (src.layout.dtype.enumv()) {
#define cb(name, ctype)                                               \
    case (DTypeEnum::name): {                                         \
        ctype* sptr = src.compatible_ptr<ctype>();                    \
        ctype* dptr = dst.compatible_ptr<ctype>();                    \
        padding_to_workspace<ctype>(dptr, sptr, src.layout, pad_axis, \
                                    align_size);                      \
        break;                                                        \
    }

        cb(Float32, dt_float32);
        default:
            megdnn_assert(0);
#undef cb
    }
}

template <typename dtype>
void padding_filter_to_workspace(dtype* dptr, const dtype* sptr, size_t OC,
                                 size_t IC, size_t FH, size_t FW) {
    size_t IC4 = (IC + 3) / 4 * 4;
    size_t HW = FH * FW;
    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t ic = 0; ic < IC4; ++ic) {
            for (size_t hw = 0; hw < HW; ++hw) {
                if (ic < IC) {
                    *dptr = sptr[oc * IC * HW + ic * HW + hw];
                } else {
                    *dptr = 0;
                }
                dptr++;
            }
        }
    }
}
}  // anonymous namespace

size_t RelayoutFormatImpl::get_workspace_in_bytes(const TensorLayout& src,
                                                  const TensorLayout& dst) {
    using Param = param::RelayoutFormat;
    switch (param().mode) {
        case Param::Mode::NCHW_NHWCD4I: {
            if (src[1] % 4 == 0)
                return 0;
            size_t IC4 = dst[2] * 4;
            size_t N = src[0];
            size_t IH = src[2];
            size_t IW = src[3];
            return N * IC4 * IH * IW * src.dtype.size();
        }
        case Param::Mode::INTER_WEIGHT_DENSEI_DOT: {
            if (src[1] % 4 == 0)
                return 0;
            size_t OC = src[0];
            size_t IC4 = dst[3] * 4;
            size_t FH = src[2];
            size_t FW = src[3];
            megdnn_assert(!(OC & 0x3));
            return OC * IC4 * FH * FW;
        }
        case Param::Mode::NCHW_NCHW88: {
            if (src[1] % 8 == 0)
                return 0;
            size_t n = src[0];
            size_t c = round_up(src[1], 8_z);
            size_t h = src[2];
            size_t w = src[3];
            return n * c * h * w * src.dtype.size();
        }
        case Param::Mode::NCHW_NCHW88_CONV_DENSE_WEIGHT: {
            megdnn_assert(src.ndim == 4, "src must be oihw ,nmdim == 5");
            megdnn_assert(src[0] % 8 == 0,
                          "NCHW_NCHW88_CONV_DENSE_WEIGHT oc must align to 8");
            if (src[1] % 8 == 0)
                return 0;
            size_t oc = src[0];
            size_t ic = round_up(src[1], 8_z);
            size_t h = src[2];
            size_t w = src[3];
            return oc * ic * h * w * src.dtype.size();
        }
        case Param::Mode::NCHW_NCHW88_CONV_GROUP_WEIGHT: {
            megdnn_assert(src.ndim == 5, "src must be goihw ,nmdim == 5");
            megdnn_assert(src[1] % 8 == 0,
                          "NCHW_NCHW88_CONV_CHAN_WEIGHT oc per group must "
                          "align to 8");
            if (src[2] % 8 == 0)
                return 0;
            size_t group = src[0];
            size_t ocpg = src[1];
            size_t icpg = round_up(src[2], 8_z);
            size_t h = src[3];
            size_t w = src[4];
            return group * ocpg * icpg * h * w * src.dtype.size();
        }
        case Param::Mode::NCHW_NCHW88_CONV_CHAN_WEIGHT: {
            megdnn_assert(src.ndim == 5, "src must be goihw ,nmdim == 5");
            if (src[0] % 8 == 0)
                return 0;
            size_t group = round_up(src[0], 8_z);
            size_t ocpg = src[1];
            size_t icpg = src[2];
            size_t h = src[3];
            size_t w = src[4];
            return group * ocpg * icpg * h * w * src.dtype.size();
        }
        default:
            return 0;
    }
}

void RelayoutFormatImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                              _megdnn_workspace workspace) {
    megdnn_assert(src.layout.dtype.category() == DTypeCategory::FLOAT ||
                  src.layout.dtype.category() == DTypeCategory::QUANTIZED);
    check_exec(src.layout, dst.layout, workspace.size);
    HandleImpl* m_handle = static_cast<HandleImpl*>(handle());
    TensorLayout exec_src, exec_dst;
    deduce_exec_layout(src.layout, dst.layout, exec_src, exec_dst);
    TensorND exec_src_nd{src.raw_ptr, exec_src};
    TensorND exec_dst_nd{dst.raw_ptr, exec_dst};
    // clean dst
    MEGDNN_DISPATCH_CPU_KERN(
            m_handle, memset(dst.raw_ptr, 0, dst.layout.span().dist_byte()));
    if (param().mode == Param::Mode::NCHW_NHWCD4I) {
        size_t N = src.layout[0];
        size_t IC = src.layout[1];
        size_t IH = src.layout[2];
        size_t IW = src.layout[3];
        //! ic % 4 != 0
        if ((IC & 0x3)) {
            switch (src.layout.dtype.enumv()) {
#define cb(name, ctype)                                                       \
    case (DTypeEnum::name): {                                                 \
        ctype* sptr = src.compatible_ptr<ctype>();                            \
        ctype* dptr = workspace.ptr<ctype>();                                 \
        MEGDNN_DISPATCH_CPU_KERN(                                             \
                m_handle,                                                     \
                padding_src_to_workspace<ctype>(dptr, sptr, N, IC, IH, IW);); \
        break;                                                                \
    }
                cb(Float32, dt_float32);
                MEGDNN_INC_FLOAT16(cb(Float16, dt_float16));
                cb(Quantized8Asymm, dt_uint8);
                cb(QuantizedS8, dt_int8);
#undef cb
                default:
                    megdnn_assert(0);
            }
            exec_src_nd.raw_ptr = workspace.raw_ptr;
        }
    } else if (param().mode == Param::Mode::INTER_WEIGHT_DENSEI_DOT) {
        size_t OC = src.layout[0];
        size_t IC = src.layout[1];
        size_t FH = src.layout[2];
        size_t FW = src.layout[3];
        if ((IC & 0x3)) {
            switch (src.layout.dtype.enumv()) {
#define cb(name, ctype)                                                      \
    case (DTypeEnum::name): {                                                \
        ctype* sptr = src.compatible_ptr<ctype>();                           \
        ctype* dptr = workspace.ptr<ctype>();                                \
        MEGDNN_DISPATCH_CPU_KERN(                                            \
                m_handle, padding_filter_to_workspace<ctype>(dptr, sptr, OC, \
                                                             IC, FH, FW););  \
        break;                                                               \
    }
                cb(Quantized8Asymm, dt_uint8);
                cb(QuantizedS8, dt_int8);
#undef cb
                default:
                    megdnn_assert(0);
            }
            exec_src_nd.raw_ptr = workspace.raw_ptr;
        }
    } else if (param().mode == Param::Mode::NCHW_NCHW88) {
        size_t ic = src.layout[1];
        if (ic % 8 != 0) {
            padding_to_workspace({workspace.raw_ptr, exec_src}, src, 1, 8);
            exec_src_nd.raw_ptr = workspace.raw_ptr;
        }
    } else if (param().mode == Param::Mode::NCHW_NCHW88_CONV_DENSE_WEIGHT) {
        megdnn_assert(src.layout[0] % 8 == 0);
        size_t ic = src.layout[1];
        if (ic % 8 != 0) {
            padding_to_workspace({workspace.raw_ptr, exec_src}, src, 1, 8_z);
            exec_src_nd.raw_ptr = workspace.raw_ptr;
        }
    } else if (param().mode == Param::Mode::NCHW_NCHW88_CONV_CHAN_WEIGHT) {
        size_t group = src.layout[0];
        if (group % 8 != 0) {
            padding_to_workspace({workspace.raw_ptr, exec_src}, src, 0, 8_z);
            exec_src_nd.raw_ptr = workspace.raw_ptr;
        }
    } else if (param().mode == Param::Mode::NCHW_NCHW88_CONV_GROUP_WEIGHT) {
        megdnn_assert(src.layout[1] % 8 == 0);
        size_t ic = src.layout[2];
        if (ic % 8 != 0) {
            padding_to_workspace({workspace.raw_ptr, exec_src}, src, 2, 8_z);
            exec_src_nd.raw_ptr = workspace.raw_ptr;
        }
    }
    m_handle->relayout_opr()->exec(exec_src_nd, exec_dst_nd, handle());
}

// vim: syntax=cpp.doxygen
