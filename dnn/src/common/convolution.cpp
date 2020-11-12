/**
 * \file dnn/src/common/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

using namespace megdnn;

namespace {
template <typename Param>
std::string get_errmsg(const TensorLayout& src, const TensorLayout& filter,
                       const TensorLayout& dst, const Param& param) {
    MEGDNN_MARK_USED_VAR(src);
    MEGDNN_MARK_USED_VAR(filter);
    MEGDNN_MARK_USED_VAR(dst);
    return megdnn_layout_msg(src) + ", " + megdnn_layout_msg(filter) + ", " +
           megdnn_layout_msg(dst) + ", " + megdnn_mangle("is_nchw=") +
           std::to_string(param.format == param::Convolution::Format::NCHW) +
           ", " + +megdnn_mangle("is_xcorr=") +
           std::to_string(
                   (param.mode == Convolution::Mode::CROSS_CORRELATION)) +
           ", " + megdnn_mangle("pad_h=") + std::to_string(param.pad_h) + ", " +
           megdnn_mangle("pad_w=") + std::to_string(param.pad_w) + ", " +
           megdnn_mangle("stride_h=") + std::to_string(param.stride_h) + ", " +
           megdnn_mangle("stride_w=") + std::to_string(param.stride_w) + ", " +
           megdnn_mangle("dilate_h=") + std::to_string(param.dilate_h) + ", " +
           megdnn_mangle("dilate_w=") + std::to_string(param.dilate_w);
}

template <typename Param, typename Param::Format>
uint32_t spatial_getter(uint32_t filter, const Param&) {
    return filter;
}

template <>
uint32_t
spatial_getter<param::ConvBias, param::ConvBias::Format::NCHW_WINOGRAD>(
        uint32_t filter, const param::ConvBias& param) {
    //! f = m + r - 1 -> r = f + 1 - m
    return filter - param.output_block_size + 1;
}

template <>
uint32_t
spatial_getter<param::ConvBias, param::ConvBias::Format::NCHW88_WINOGRAD>(
        uint32_t filter, const param::ConvBias& param) {
    //! f = m + r - 1 -> r = f + 1 - m
    return filter - param.output_block_size + 1;
}
template <>
uint32_t
spatial_getter<param::ConvBias, param::ConvBias::Format::NCHW44_WINOGRAD>(
        uint32_t filter, const param::ConvBias& param) {
    //! f = m + r - 1 -> r = f + 1 - m
    return filter - param.output_block_size + 1;
}

template <typename Parameter, typename Param>
void make_canonized_filter_meta_nchw_nhwc(
        size_t src_ndim, const TensorLayout& filter, const Param& param,
        typename ConvolutionBase<Parameter>::CanonizedFilterMeta& ret) {
    megdnn_assert(param.format == Param::Format::NCHW ||
                  param.format == Param::Format::NHWC ||
                  param.format == Param::Format::NCHW_WINOGRAD);
    auto img_ndim = src_ndim - 2;
    size_t flt_start, flt_spatial_start, ocpg_pos, icpg_pos;
    if (param.sparse == Param::Sparse::DENSE) {
        megdnn_assert(
                filter.ndim == img_ndim + 2 || filter.ndim == img_ndim + 4,
                "bad filter ndim for dense convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);
        ret.group = 1;
        flt_start = 0;
    } else {
        megdnn_assert(param.sparse == Param::Sparse::GROUP,
                      "invalid convolution sparse type");
        megdnn_assert(
                filter.ndim == img_ndim + 3 || filter.ndim == img_ndim + 5,
                "bad filter ndim for group convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);

        // grp, oc, ic, dims[]
        ret.group = filter[0];
        flt_start = 1;
    }

    uint32_t ic_block_size = 1, oc_block_size = 1;
    if (param.format == Param::Format::NCHW) {
        // filter should be (oc, ic, fh, fw)
        flt_spatial_start = 2;
        ocpg_pos = 0;
        icpg_pos = 1;
    } else if (param.format == Param::Format::NCHW_WINOGRAD) {
        // filter should be (alphah, alphaw, ic, oc) or (alphah, alphaw, ocb,
        // icb, ic_block_size, oc_block_size)
        flt_spatial_start = 0;
        if (filter.ndim == flt_start + 4) {
            ocpg_pos = 3;
            icpg_pos = 2;
        } else {
            megdnn_assert(filter.ndim == flt_start + 6);
            ic_block_size = filter[flt_start + 4];
            oc_block_size = filter[flt_start + 5];
            ocpg_pos = 2;
            icpg_pos = 3;
        }
    } else {
        megdnn_assert(param.format == Param::Format::NHWC,
                      "invalid conv tensor format");
        // filter should be (oc, fh, fw, ic)
        flt_spatial_start = 1;
        ocpg_pos = 0;
        icpg_pos = 3;
    }
    ret.spatial_ndim = src_ndim - 2;
    megdnn_assert(
            ret.spatial_ndim == 2,
            "only 2D convolution is supported, and input should be 4-dim; "
            "got input dim = %zu",
            src_ndim);
    ret.ocpg = filter[flt_start + ocpg_pos] * oc_block_size;
    ret.icpg = filter[flt_start + icpg_pos] * ic_block_size;
    auto dilation = ret.dilation;
    for (size_t i = 0; i < ret.spatial_ndim; ++i) {
        megdnn_assert(dilation[i] > 0,
                      "invalid dilation on spatial dim %zu: %u", i,
                      dilation[i]);
        if (param.format == Param::Format::NCHW_WINOGRAD) {
            ret.spatial[i] =
                    spatial_getter<Param, Param::Format::NCHW_WINOGRAD>(
                            filter[i + flt_start + flt_spatial_start], param);
        } else {
            ret.spatial[i] = spatial_getter<Param, Param::Format::NCHW>(
                    filter[i + flt_start + flt_spatial_start], param);
        }
        ret.dilated_spatial[i] = (ret.spatial[i] - 1) * dilation[i] + 1;
    }
}

template <typename Parameter, typename Param>
void make_canonized_filter_meta_nhwcd4(
        size_t src_ndim, const TensorLayout& filter, const Param& param,
        typename ConvolutionBase<Parameter>::CanonizedFilterMeta& ret) {
    /**
     * input: N H IC/4 W 4
     * Filter:
     *        OC/4, FH, FW, IC, 4 [dense]
     *        GROUP, OC/4, FH, FW, IC, 4 [group]
     *        GROUP/4, 1, FH, FW, 4 [chanwise]
     */
    megdnn_assert(param.format == Param::Format::NHWCD4);
    auto img_ndim = src_ndim - 3;
    size_t flt_start = 0, flt_spatial_start = 1;
    bool is_chanwise = false;
    if (param.sparse == Param::Sparse::DENSE) {
        megdnn_assert(filter.ndim == img_ndim + 3,
                      "bad filter ndim for dense convolution: "
                      "spatial_ndim=%zu filter_ndim=%zu",
                      img_ndim, filter.ndim);
        // oc, ic, dims[]
        ret.group = 1;
        flt_start = 0;
    } else {
        megdnn_assert(param.sparse == Param::Sparse::GROUP,
                      "invalid convolution sparse type");
        megdnn_assert(
                filter.ndim == img_ndim + 3 || filter.ndim == img_ndim + 4,
                "bad filter ndim for group convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);
        if (filter.ndim == img_ndim + 3 && filter[1] == 1) {
            is_chanwise = true;
            ret.group = filter[0] * 4;
        } else {
            ret.group = filter[0];
        }
        flt_start = 1;
    }
    ret.spatial_ndim = src_ndim - 3;
    megdnn_assert(
            ret.spatial_ndim == 2,
            "only 2D convolution is supported, and input should be 4-dim; "
            "got input dim = %zu",
            src_ndim);
    if (is_chanwise) {
        ret.ocpg = 1;
        ret.icpg = 1;
    } else {
        ret.ocpg = filter[flt_start] * 4;
        ret.icpg = filter[flt_start + 3];
    }
    auto dilation = ret.dilation;
    for (size_t i = 0; i < ret.spatial_ndim; ++i) {
        megdnn_assert(dilation[i] > 0,
                      "invalid dilation on spatial dim %zu: %u", i,
                      dilation[i]);
        ret.spatial[i] = filter[i + flt_start + flt_spatial_start];
        ret.dilated_spatial[i] = (ret.spatial[i] - 1) * dilation[i] + 1;
    }
}

template <typename Parameter, typename Param>
void make_canonized_filter_meta_nhwcd4_dot(
        size_t src_ndim, const TensorLayout& filter, const Param& param,
        typename ConvolutionBase<Parameter>::CanonizedFilterMeta& ret) {
    /**
     * input: N H IC/4 W 4
     * Filter:
     *        GROUP/4, 1, FH, FW, 4 [chanwise]
     *        OC/4, FH, FW, IC/4, 4, 4 [dense]
     *        GROUP, OC/4, FH, FW, IC/4, 4, 4 [group]
     */
    megdnn_assert(param.format == Param::Format::NHWCD4);
    auto img_ndim = src_ndim - 3;
    size_t flt_start = 0, flt_spatial_start = 1;
    bool is_chanwise = false;
    if (param.sparse == Param::Sparse::DENSE) {
        megdnn_assert(filter.ndim == img_ndim + 4,
                      "bad filter ndim for dense convolution: "
                      "spatial_ndim=%zu filter_ndim=%zu",
                      img_ndim, filter.ndim);
        // oc, ic, dims[]
        ret.group = 1;
        flt_start = 0;
    } else {
        megdnn_assert(param.sparse == Param::Sparse::GROUP,
                      "invalid convolution sparse type");
        megdnn_assert(
                filter.ndim == img_ndim + 3 || filter.ndim == img_ndim + 5,
                "bad filter ndim for group convolution: "
                "spatial_ndim=%zu filter_ndim=%zu",
                img_ndim, filter.ndim);
        if (filter.ndim == img_ndim + 3) {
            megdnn_assert(filter[1] == 1);
            is_chanwise = true;
            ret.group = filter[0] * 4;
        } else {
            ret.group = filter[0];
        }
        flt_start = 1;
    }
    ret.spatial_ndim = src_ndim - 3;
    megdnn_assert(
            ret.spatial_ndim == 2,
            "only 2D convolution is supported, and input should be 4-dim; "
            "got input dim = %zu",
            src_ndim);
    if (is_chanwise) {
        ret.ocpg = 1;
        ret.icpg = 1;
    } else {
        ret.ocpg = filter[flt_start] * 4;
        ret.icpg = filter[flt_start + 3] * 4;
    }
    auto dilation = ret.dilation;
    for (size_t i = 0; i < ret.spatial_ndim; ++i) {
        megdnn_assert(dilation[i] > 0,
                      "invalid dilation on spatial dim %zu: %u", i,
                      dilation[i]);
        ret.spatial[i] = filter[i + flt_start + flt_spatial_start];
        ret.dilated_spatial[i] = (ret.spatial[i] - 1) * dilation[i] + 1;
    }
}

template <size_t pack_size, typename Parameter, typename Param>
void make_canonized_filter_meta_nchwxx(
        size_t src_ndim, const TensorLayout& filter, const Param& param,
        typename ConvolutionBase<Parameter>::CanonizedFilterMeta& ret) {
    /**
     * input: N IC/pack_size, H, W, pack_size
     *
     ** NCHW44-DOT mode
     * filter:
     *        {OC/pack_size, IC/pack_size, FH, FW, pack_size(OC), pack_size(IC)}
     * [dense]
     *        {GROUP, OC_PER_GROUP/pack_size, IC_PER_GROUP/pack_size, \
     *                  FH, FW, pack_size(OC), pack_size(IC)} [group]
     *
     * NCHW88 and NCHW44 mode
     * filter:
     *        {OC/pack_size, IC/pack_size, FH, FW, pack_size(IC), pack_size(OC)}
     * [dense]
     *        {GROUP, OC_PER_GROUP/pack_size, IC_PER_GROUP/pack_size, \
     *                  FH, FW, pack_size(IC), pack_size(OC)} [group]
     *        {GROUP/pack_size, 1, 1, FH, FW, pack_size} [chan]
     *
     ** NCHW88_WINOGRAD and NCHW44_WINOGRAD mode
     * filter:
     *        {alpha, alpha, OC/pack_size, IC/pack_size, pack_size(IC),
     *pack_size(OC)} [dense]
     *        {GROUP, alpha, alpha, OC_PER_GROUP/pack_size,
     *          IC_PER_GROUP/pack_size, pack_size(IC), pack_size(OC)} [group]
     *
     */

    megdnn_assert(param.format == Param::Format::NCHW88 ||
                  param.format == Param::Format::NCHW44 ||
                  param.format == Param::Format::NCHW44_WINOGRAD ||
                  param.format == Param::Format::NCHW44_DOT ||
                  param.format == Param::Format::NCHW88_WINOGRAD);
    size_t img_ndim = 2;
    size_t flt_start = 0;
    size_t flt_spatial_start = 2;
    size_t pack_c_size = 0;
    if (param.sparse == Param::Sparse::DENSE) {
        if (filter.ndim == img_ndim + 4) {
            // oihw8i8o case
            megdnn_assert((filter[filter.ndim - 2] == pack_size &&
                           filter[filter.ndim - 1] == pack_size) ||
                                  (filter[filter.ndim - 2] == 2 * pack_size &&
                                   filter[filter.ndim - 1] == 2 * pack_size),
                          "last 2 dim of filter must be %zu, but got %zu, %zu",
                          pack_size, filter[filter.ndim - 2],
                          filter[filter.ndim - 1]);
            ret.group = 1;
            flt_start = 0;
            if (param.format == Param::Format::NCHW88_WINOGRAD ||
                param.format == Param::Format::NCHW44_WINOGRAD) {
                flt_start = 2;
            }
            if (filter[filter.ndim - 2] == 2 * pack_size &&
                filter[filter.ndim - 1] == 2 * pack_size) {
                pack_c_size = 2 * pack_size;
            } else {
                pack_c_size = pack_size;
            }
            ret.ocpg = filter[flt_start] * pack_c_size;
            ret.icpg = filter[flt_start + 1] * pack_c_size;
        } else if (filter.ndim == img_ndim + 3) {
            // ohwi8o
            megdnn_assert(param.format != Param::Format::NCHW88_WINOGRAD,
                          "Hybrid nchw88 mode in not support winograd");
            megdnn_assert(param.format != Param::Format::NCHW44_WINOGRAD,
                          "Hybrid nchw44 mode in not support winograd");
            flt_start = 0;
            flt_spatial_start = 1;
            ret.group = 1;
            ret.ocpg = filter[flt_start] * pack_size;
            ret.icpg = filter[flt_start + 3];

        } else {
            megdnn_assert(0, "not support nchwxx filter dim = %zu",
                          filter.ndim);
        }
    } else {
        megdnn_assert(param.sparse == Param::Sparse::GROUP,
                      "invalid convolution sparse type");
        flt_start = 1;
        if (param.format == Param::Format::NCHW88_WINOGRAD ||
            param.format == Param::Format::NCHW44_WINOGRAD) {
            flt_start = 3;
        }
        auto filter_oc = filter[flt_start];
        auto filter_ic = filter[flt_start + 1];
        if (filter_oc == 1 && filter_ic == 1 && filter.ndim == (img_ndim + 4) &&
            param.format != Param::Format::NCHW88_WINOGRAD &&
            param.format != Param::Format::NCHW44_WINOGRAD) {
            // Depthwise case goihw8g
            megdnn_assert(filter.ndim == img_ndim + 4,
                          "bad filter ndim for group convolution: "
                          "spatial_ndim=%zu filter_ndim=%zu",
                          img_ndim, filter.ndim);
            megdnn_assert(filter[filter.ndim - 1] == pack_size,
                          "last dim of filter must be %zu, but %zu", pack_size,
                          filter[filter.ndim - 1]);
            ret.group = filter[0] * pack_size;
            ret.ocpg = filter_oc;
            ret.icpg = filter_ic;

        } else {
            // norm group case goihw8i8o
            megdnn_assert(filter.ndim == img_ndim + 5,
                          "bad filter ndim for group convolution: "
                          "spatial_ndim=%zu filter_ndim=%zu",
                          img_ndim, filter.ndim);
            megdnn_assert((filter[filter.ndim - 1] == pack_size &&
                           filter[filter.ndim - 2] == pack_size) ||
                           (filter[filter.ndim - 1] == 2 * pack_size &&
                            filter[filter.ndim - 2] == 2 * pack_size),
                          "last 2 dim of filter must be %zu, but got %zu, %zu",
                          pack_size, filter[filter.ndim - 2],
                          filter[filter.ndim - 1]);

            ret.group = filter[0];
            if (filter[filter.ndim - 2] == 2 * pack_size &&
                filter[filter.ndim - 1] == 2 * pack_size) {
                ret.ocpg = filter_oc * 2 * pack_size;
                ret.icpg = filter_ic * 2 * pack_size;
            } else {
                ret.ocpg = filter_oc * pack_size;
                ret.icpg = filter_ic * pack_size;
            }
        }
    }
    ret.spatial_ndim = 2;
    megdnn_assert(ret.spatial_ndim == 2,
                  "only 2D convolution is supported, and input should be 5-dim "
                  "for nchwxx; "
                  "got input dim = %zu",
                  src_ndim);

    auto dilation = ret.dilation;
    for (size_t i = 0; i < ret.spatial_ndim; ++i) {
        megdnn_assert(dilation[i] == 1,
                      "NCHWXX has invalid dilation on spatial dim %zu: %u, "
                      "require to be 1",
                      i, dilation[i]);
        if (param.format == Param::Format::NCHW88_WINOGRAD) {
            ret.spatial[i] =
                    spatial_getter<Param, Param::Format::NCHW88_WINOGRAD>(
                            filter[i + flt_start - 2], param);
        } else if (param.format == Param::Format::NCHW44_WINOGRAD) {
            ret.spatial[i] =
                    spatial_getter<Param, Param::Format::NCHW44_WINOGRAD>(
                            filter[i + flt_start - 2], param);
        } else {
            ret.spatial[i] = filter[i + flt_start + flt_spatial_start];
        }
        ret.dilated_spatial[i] = (ret.spatial[i] - 1) * dilation[i] + 1;
    }
}

template <size_t pack_size, typename Parameter, typename Param>
void make_canonized_filter_meta_nchwx(
        size_t src_ndim, const TensorLayout& filter, const Param& param,
        typename ConvolutionBase<Parameter>::CanonizedFilterMeta& ret) {
    /**
     * input: N IC/pack_size, H, W, pack_size
     * filter:
     *        OC, IC/pack_size, FH, FW, pack_size [dense]
     *        GROUP, OC, IC/pack_size, FH, FW, pack_size [group]
     */
    megdnn_assert(param.format == Param::Format::NCHW4 ||
                  param.format == Param::Format::NCHW8 ||
                  param.format == Param::Format::NCHW32 ||
                  param.format == Param::Format::NCHW4_NCHW ||
                  param.format == Param::Format::NCHW4_NCHW32 ||
                  param.format == Param::Format::NCHW32_NCHW4);
    auto img_ndim = src_ndim - 3;
    size_t flt_start = 0, flt_spatial_start = 2;
    if (param.sparse == Param::Sparse::DENSE) {
        megdnn_assert(filter.ndim == img_ndim + 3,
                      "bad filter ndim for dense convolution: "
                      "spatial_ndim=%zu filter_ndim=%zu",
                      img_ndim, filter.ndim);
        // oc, ic, dims[]
        ret.group = 1;
        flt_start = 0;
    } else {
        megdnn_assert(param.sparse == Param::Sparse::GROUP,
                      "invalid convolution sparse type");
        megdnn_assert(filter.ndim == img_ndim + 4,
                      "bad filter ndim for group convolution: "
                      "spatial_ndim=%zu filter_ndim=%zu",
                      img_ndim, filter.ndim);
        ret.group = filter[0];
        flt_start = 1;
    }
    ret.spatial_ndim = src_ndim - 3;
    megdnn_assert(ret.spatial_ndim == 2,
                  "only 2D convolution is supported, and input should be 5-dim "
                  "for nchw4; "
                  "got input dim = %zu",
                  src_ndim);
    ret.ocpg = filter[flt_start];
    ret.icpg = filter[flt_start + 1] * pack_size;
    auto dilation = ret.dilation;
    for (size_t i = 0; i < ret.spatial_ndim; ++i) {
        megdnn_assert(dilation[i] == 1,
                      "NCHW4 has invalid dilation on spatial dim %zu: %u, "
                      "require to be 1",
                      i, dilation[i]);
        ret.spatial[i] = filter[i + flt_start + flt_spatial_start];
        ret.dilated_spatial[i] = (ret.spatial[i] - 1) * dilation[i] + 1;
    }
}

template <size_t pack_size, typename Parameter, typename Param>
void make_canonized_filter_meta_chwnx(
        size_t src_ndim, const TensorLayout& filter, const Param& param,
        typename ConvolutionBase<Parameter>::CanonizedFilterMeta& ret) {
    /**
     * input: IC / pack_size, H, W, N, pack_size
     * Filter:
     *        IC / pack_size, FH, FW, OC, pack_size [dense]
     *        GROUP, icpg / pack_size, FH, FW, ocpg, pack_size [group]
     *        not implemented [chanwise]
     */
    megdnn_assert(param.format == Param::Format::CHWN4);
    auto img_ndim = src_ndim - 3;
    size_t flt_start = 0, flt_spatial_start = 1;
    if (param.sparse == Param::Sparse::DENSE) {
        megdnn_assert(filter.ndim == img_ndim + 3,
                      "bad filter ndim for dense convolution: "
                      "spatial_ndim=%zu filter_ndim=%zu",
                      img_ndim, filter.ndim);
        // oc, ic, dims[]
        ret.group = 1;
        flt_start = 0;
    } else {
        megdnn_assert(param.sparse == Param::Sparse::GROUP,
                      "invalid convolution sparse type");
        megdnn_assert(filter.ndim == img_ndim + 4,
                      "bad filter ndim for group convolution: "
                      "spatial_ndim=%zu filter_ndim=%zu",
                      img_ndim, filter.ndim);
        ret.group = filter[0];
        flt_start = 1;
    }
    ret.spatial_ndim = src_ndim - 3;
    megdnn_assert(
            ret.spatial_ndim == 2,
            "only 2D convolution is supported, and input should be 4-dim; "
            "got input dim = %zu",
            src_ndim);
    ret.icpg = filter[flt_start] * pack_size;
    ret.ocpg = filter[flt_start + 3];
    auto dilation = ret.dilation;
    for (size_t i = 0; i < ret.spatial_ndim; ++i) {
        megdnn_assert(dilation[i] == 1,
                      "CHWNx has invalid dilation on spatial dim %zu: %u, "
                      "require to be 1",
                      i, dilation[i]);
        ret.spatial[i] = filter[i + flt_start + flt_spatial_start];
        ret.dilated_spatial[i] = (ret.spatial[i] - 1) * dilation[i] + 1;
    }
}

}  // namespace

namespace megdnn {
template <typename Parameter>
typename ConvolutionBase<Parameter>::CanonizedFilterMeta
ConvolutionBase<Parameter>::make_canonized_filter_meta(
        size_t src_ndim, const TensorLayout& filter) const {
    megdnn_assert_contiguous(filter);
    CanonizedFilterMeta ret;
    ret.dtype = filter.dtype;
    ret.format = param().format;
    if (param().mode == Mode::CONVOLUTION) {
        ret.should_flip = true;
    } else {
        megdnn_assert(param().mode == Mode::CROSS_CORRELATION,
                      "invalid conv mode");
        ret.should_flip = false;
    }
    ret.stride[0] = param().stride_h;
    ret.stride[1] = param().stride_w;
    ret.padding[0] = param().pad_h;
    ret.padding[1] = param().pad_w;
    ret.dilation[0] = param().dilate_h;
    ret.dilation[1] = param().dilate_w;

    if (param().format == Param::Format::NHWCD4) {
        if (filter.dtype.enumv() == DTypeEnum::QuantizedS8 ||
            filter.dtype.enumv() == DTypeEnum::Quantized8Asymm) {
            make_canonized_filter_meta_nhwcd4_dot<Parameter>(src_ndim, filter,
                                                             param(), ret);
        } else {
            make_canonized_filter_meta_nhwcd4<Parameter>(src_ndim, filter,
                                                         param(), ret);
        }
    } else if (param().format == Param::Format::NCHW4 ||
               param().format == Param::Format::NCHW4_NCHW ||
               param().format == Param::Format::NCHW4_NCHW32) {
        make_canonized_filter_meta_nchwx<4, Parameter>(src_ndim, filter,
                                                       param(), ret);
    } else if (param().format == Param::Format::NCHW8) {
        make_canonized_filter_meta_nchwx<8, Parameter>(src_ndim, filter,
                                                       param(), ret);
    } else if (param().format == Param::Format::NCHW88 ||
               param().format == Param::Format::NCHW88_WINOGRAD) {
        make_canonized_filter_meta_nchwxx<8, Parameter>(src_ndim, filter,
                                                        param(), ret);
    } else if (param().format == Param::Format::NCHW44 ||
               param().format == Param::Format::NCHW44_DOT ||
               param().format == Param::Format::NCHW44_WINOGRAD) {
        make_canonized_filter_meta_nchwxx<4, Parameter>(src_ndim, filter,
                                                        param(), ret);
    } else if (param().format == Param::Format::NCHW32 ||
               param().format == Param::Format::NCHW32_NCHW4) {
        make_canonized_filter_meta_nchwx<32, Parameter>(src_ndim, filter,
                                                        param(), ret);
    } else if (param().format == Param::Format::CHWN4) {
        make_canonized_filter_meta_chwnx<4, Parameter>(src_ndim, filter,
                                                       param(), ret);
    } else {
        megdnn_assert(param().format == Param::Format::NHWC ||
                      param().format == Param::Format::NCHW ||
                      param().format == Param::Format::NCHW_WINOGRAD);
        make_canonized_filter_meta_nchw_nhwc<Parameter>(src_ndim, filter,
                                                        param(), ret);
    }
    return ret;
}

template <typename Parameter>
void ConvolutionBase<Parameter>::check_or_deduce_dtype_fwd(DType src,
                                                           DType filter,
                                                           DType& dst) const {
    // The first one will be the default choice.
    SmallVector<DType> supported_dst_dtype;
    // We rely on megdnn_assert(src.enumv() == filter.enumv()) here.
    if (src.category() == DTypeCategory::FLOAT) {
        supported_dst_dtype.push_back(src);
    } else if (src.enumv() == DTypeEnum::Int8) {
        supported_dst_dtype = {dtype::Int32(), dtype::Int16()};
    } else if (src.enumv() == DTypeEnum::QuantizedS8 ||
               src.enumv() == DTypeEnum::Quantized8Asymm ||
               src.enumv() == DTypeEnum::Quantized4Asymm) {
        //! Qint8 winograd compute with float, in order to bringing the filter
        //! scale, here just use QuantizedS32 as filter type.
        if (src.enumv() == DTypeEnum::QuantizedS8 &&
            filter.enumv() == DTypeEnum::QuantizedS32) {
            supported_dst_dtype.push_back(dtype::QuantizedS32(
                    src.param<dtype::QuantizedS8>().scale *
                    filter.param<dtype::QuantizedS32>().scale));
        } else {
            supported_dst_dtype.push_back(
                    dtype::QuantizedS32(mul_scale(src, filter)));
        }
        if (dst.valid() && dst.enumv() == src.enumv()) {
            supported_dst_dtype.push_back(dst);
        }
        if (src.enumv() == DTypeEnum::QuantizedS8) {
            supported_dst_dtype.push_back(dtype::Float32());
        }
    } else if (src.enumv() == DTypeEnum::QuantizedS32) {
        //! ConvolutionBackwardData: s8(filter) + s8(dst) -> s32(src)
        megdnn_assert(filter.enumv() == DTypeEnum::QuantizedS8);
        supported_dst_dtype.push_back(
                dtype::QuantizedS8(src.param<dtype::QuantizedS32>().scale /
                                   filter.param<dtype::QuantizedS8>().scale));
    } else {
        megdnn_throw(ssprintf("unsupported input / filter DType: %s x %s",
                              src.name(), filter.name()));
    }
    if (!dst.valid()) {
        dst = supported_dst_dtype.at(0);
    } else {
        bool dst_supported = false;
        for (auto&& dt : supported_dst_dtype) {
            if (dtype_almost_equal(dt, dst)) {
                dst_supported = true;
                break;
            }
        }
        MEGDNN_MARK_USED_VAR(dst_supported);
        megdnn_assert(dst_supported, "unsupported Conv(%s, %s) -> %s",
                      src.name(), filter.name(), dst.name());
    }
    megdnn_assert((param().compute_mode == Param::ComputeMode::FLOAT32 ||
                   param().compute_mode == Param::ComputeMode::DEFAULT)
#if !MEGDNN_DISABLE_FLOAT16
                          || src.enumv() == DTypeEnum::Float16 ||
                          src.enumv() == DTypeEnum::BFloat16
#endif
                  ,
                  "ComputeMode::FLOAT32 is only available for Float16/BFloat16 "
                  "input / output.");
}

template <typename Parameter>
typename ConvolutionBase<Parameter>::CanonizedFilterMeta
ConvolutionBase<Parameter>::deduce_layout_fwd(const TensorLayout& src,
                                              const TensorLayout& filter,
                                              TensorLayout& dst) const {
    auto errmsg = [&]() { return get_errmsg(src, filter, dst, param()); };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(filter);
    megdnn_assert(src.ndim >= 3_z, "%s", errmsg().c_str());
    if ((param().format == Param::Format::NCHW_WINOGRAD ||
         param().format == Param::Format::NCHW44_WINOGRAD) &&
        src.dtype.category() == DTypeCategory::QUANTIZED) {
        megdnn_assert((filter.dtype.enumv() == DTypeEnum::QuantizedS16 ||
                       filter.dtype.enumv() == DTypeEnum::QuantizedS32),
                      "%s", errmsg().c_str());
        megdnn_assert(src.dtype.enumv() == DTypeEnum::QuantizedS8 ||
                              src.dtype.enumv() == DTypeEnum::Quantized8Asymm,
                      "%s", errmsg().c_str());
    } else {
        megdnn_assert(src.dtype.enumv() == filter.dtype.enumv(), "%s",
                      errmsg().c_str());
    }
    check_or_deduce_dtype_fwd(src.dtype, filter.dtype, dst.dtype);
    size_t img_dim;
    if (param().format == Param::Format::NCHW ||
        param().format == Param::Format::NHWC ||
        param().format == Param::Format::NCHW_WINOGRAD) {
        img_dim = src.ndim - 2;
        megdnn_assert(filter.ndim >= img_dim + 2 && filter.ndim <= img_dim + 6,
                      "%s", errmsg().c_str());

    } else {
        megdnn_assert(param().format == Param::Format::NHWCD4 ||
                      param().format == Param::Format::NCHW4 ||
                      param().format == Param::Format::NCHW4_NCHW ||
                      param().format == Param::Format::NCHW4_NCHW32 ||
                      param().format == Param::Format::NCHW44 ||
                      param().format == Param::Format::NCHW44_DOT ||
                      param().format == Param::Format::NCHW8 ||
                      param().format == Param::Format::NCHW32 ||
                      param().format == Param::Format::NCHW32_NCHW4 ||
                      param().format == Param::Format::NCHW88 ||
                      param().format == Param::Format::NCHW88_WINOGRAD ||
                      param().format == Param::Format::NCHW44_WINOGRAD ||
                      param().format == Param::Format::CHWN4);
        img_dim = src.ndim - 3;
        if ((param().format == Param::Format::NCHW88 ||
             param().format == Param::Format::NCHW44_DOT ||
             param().format == Param::Format::NCHW44) &&
            filter.ndim == 5) {
            img_dim = src.ndim - 2;
        }
        megdnn_assert(filter.ndim == img_dim + 3 ||
                              (filter.ndim == img_dim + 2 &&
                               (param().format == Param::Format::NCHW88 ||
                                param().format == Param::Format::NCHW44_DOT ||
                                param().format == Param::Format::NCHW44)) ||
                              filter.ndim == img_dim + 4 ||
                              filter.ndim == img_dim + 5,
                      "%s", errmsg().c_str());
        if (param().format == Param::Format::NCHW4 ||
            param().format == Param::Format::NCHW4_NCHW ||
            param().format == Param::Format::NCHW4_NCHW32) {
            megdnn_assert(src.ndim == 5 &&
                                  (filter.ndim == 5 || filter.ndim == 6 ||
                                   filter.ndim == 7) &&
                                  src[src.ndim - 1] == 4 &&
                                  filter[filter.ndim - 1] == 4,
                          "NCHW4/NCHW4_NCHW/NCHW4_NCHW32 require src and "
                          "filter's ndim is "
                          "5 or 6, and "
                          "last shape "
                          "is 4 "
                          "but got src %s, filter %s",
                          src.to_string().c_str(), filter.to_string().c_str());
        }
        if (param().format == Param::Format::NCHW8) {
            megdnn_assert(
                    src.ndim == 5 && (filter.ndim == 5 || filter.ndim == 6) &&
                            src[src.ndim - 1] == 8 &&
                            filter[filter.ndim - 1] == 8,
                    "NCHW8 require src and filter's ndim is 5 or 6, and last "
                    "shape is 8 "
                    "but got src %s, filter %s",
                    src.to_string().c_str(), filter.to_string().c_str());
        }
        if (param().format == Param::Format::NCHW32 ||
            param().format == Param::Format::NCHW32_NCHW4) {
            megdnn_assert(src.ndim == 5 &&
                                  (filter.ndim == 5 || filter.ndim == 6) &&
                                  src[src.ndim - 1] == 32 &&
                                  filter[filter.ndim - 1] == 32,
                          "NCHW32/NCHW32_NCHW4 require src and filter's ndim "
                          "is 5 or 6, and last "
                          "shape is 32 "
                          "but got src %s, filter %s",
                          src.to_string().c_str(), filter.to_string().c_str());
        }
        if (param().format == Param::Format::NCHW88 ||
            param().format == Param::Format::NCHW88_WINOGRAD) {
            megdnn_assert((src.ndim == 4 && filter.ndim == 5 &&
                           filter[filter.ndim - 1] == 8) ||
                                  (src.ndim == 5 &&
                                   ((filter.ndim == 6 &&
                                     filter[filter.ndim - 1] == 8) ||
                                    (filter.ndim == 7 &&
                                     filter[filter.ndim - 1] == 8 &&
                                     filter[filter.ndim - 2] == 8)) &&
                                   src[src.ndim - 1] == 8),
                          "NCHW88 require src ndim is 5 and filter's ndim is 6 "
                          ", and last shape two is 8 but got src %s, filter %s",
                          src.to_string().c_str(), filter.to_string().c_str());
        }
        if (param().format == Param::Format::NCHW44 ||
            param().format == Param::Format::NCHW44_DOT ||
            param().format == Param::Format::NCHW44_WINOGRAD) {
            //!support nchw44 filter change to 88 for int8 winogradf23_88 using MK8 mamtul
            megdnn_assert((src.ndim == 4 && filter.ndim == 5 &&
                           filter[filter.ndim - 1] == 4) ||
                                  (src.ndim == 5 &&
                                   ((filter.ndim == 6 &&
                                     (filter[filter.ndim - 1] == 4 ||
                                      filter[filter.ndim - 1] == 8)) ||
                                    (filter.ndim == 7 &&
                                     (filter[filter.ndim - 1] == 4 ||
                                      filter[filter.ndim - 1] == 8) &&
                                     (filter[filter.ndim - 2] == 4 ||
                                      filter[filter.ndim - 2] == 8))) &&
                                   src[src.ndim - 1] == 4),
                          "NCHW44 require src ndim is 5 and filter's ndim is 6 "
                          ", and last shape two is 4 but got src %s, filter %s",
                          src.to_string().c_str(), filter.to_string().c_str());
        }
        if (param().format == Param::Format::CHWN4) {
            megdnn_assert(
                    src.ndim == 5 && (filter.ndim == 5 || filter.ndim == 6) &&
                            src[src.ndim - 1] == 4 &&
                            filter[filter.ndim - 1] == 4,
                    "CHWN4 require src and filter's ndim is 5 or 6, and last "
                    "shape is 4 "
                    "but got src %s, filter %s",
                    src.to_string().c_str(), filter.to_string().c_str());
        }
    }
    megdnn_assert(img_dim == 2,
                  "currently only convolution on 2D image is supported");
    auto cflt = make_canonized_filter_meta(src.ndim, filter);
    if (param().format == Param::Format::NCHW ||
        param().format == Param::Format::NHWC ||
        param().format == Param::Format::NCHW_WINOGRAD) {
        size_t src_or_dst_c_pos = 0;
        size_t src_or_dst_spatial_start = 0;
        if (param().format == Param::Format::NCHW ||
            param().format == Param::Format::NCHW_WINOGRAD) {
            src_or_dst_c_pos = 1;
            src_or_dst_spatial_start = 2;
        } else {
            megdnn_assert(param().format == Param::Format::NHWC,
                          "invalid conv format");
            src_or_dst_c_pos = 3;
            src_or_dst_spatial_start = 1;
        }
        megdnn_assert(cflt.icpg * cflt.group == src[src_or_dst_c_pos], "%s",
                      errmsg().c_str());
        if (param().format == Param::Format::NCHW_WINOGRAD) {
            megdnn_assert(cflt.spatial[0] == cflt.spatial[1],
                          "NCHW_WINOGRAD only support conv with fh == fw");
        }
        dst.ndim = src.ndim;
        dst[0] = src[0];
        dst[src_or_dst_c_pos] = cflt.ocpg * cflt.group;
        for (size_t i = 0; i < cflt.spatial_ndim; ++i) {
            dst[i + src_or_dst_spatial_start] = infer_conv_shape(
                    src[i + src_or_dst_spatial_start], cflt.dilated_spatial[i],
                    cflt.stride[i], cflt.padding[i]);
        }
        dst.init_contiguous_stride();
    } else if (param().format == Param::Format::NCHW4) {
        megdnn_assert(src.ndim == 5,
                      "invalid src ndim for NCHW4, expected=5, got=%zu",
                      src.ndim);
        megdnn_assert(cflt.icpg * cflt.group == src[1] * 4,
                      "%s icpg=%u group=%u", errmsg().c_str(), cflt.icpg,
                      cflt.group);
        dst.ndim = src.ndim;
        dst[0] = src[0];
        auto oc = cflt.ocpg * cflt.group;
        megdnn_assert(oc % 4 == 0);
        dst[1] = oc / 4;
        dst[2] = infer_conv_shape(src[2], cflt.dilated_spatial[0],
                                  cflt.stride[0], cflt.padding[0]);
        dst[3] = infer_conv_shape(src[3], cflt.dilated_spatial[1],
                                  cflt.stride[1], cflt.padding[1]);
        dst[4] = 4;
    } else if (param().format == Param::Format::NCHW8) {
        megdnn_assert(src.ndim == 5,
                      "invalid src ndim for NCHW8, expected=5, got=%zu",
                      src.ndim);
        megdnn_assert(cflt.icpg * cflt.group == src[1] * 8,
                      "%s icpg=%u group=%u", errmsg().c_str(), cflt.icpg,
                      cflt.group);
        dst.ndim = src.ndim;
        dst[0] = src[0];
        auto oc = cflt.ocpg * cflt.group;
        megdnn_assert(oc % 8 == 0);
        dst[1] = oc / 8;
        dst[2] = infer_conv_shape(src[2], cflt.dilated_spatial[0],
                                  cflt.stride[0], cflt.padding[0]);
        dst[3] = infer_conv_shape(src[3], cflt.dilated_spatial[1],
                                  cflt.stride[1], cflt.padding[1]);
        dst[4] = 8;
    } else if (param().format == Param::Format::NCHW32) {
        megdnn_assert(src.ndim == 5,
                      "invalid src ndim for NCHW32, expected=5, got=%zu",
                      src.ndim);
        megdnn_assert(cflt.icpg * cflt.group == src[1] * 32,
                      "%s icpg=%u group=%u", errmsg().c_str(), cflt.icpg,
                      cflt.group);
        dst.ndim = src.ndim;
        dst[0] = src[0];
        auto oc = cflt.ocpg * cflt.group;
        megdnn_assert(oc % 32 == 0);
        dst[1] = oc / 32;
        dst[2] = infer_conv_shape(src[2], cflt.dilated_spatial[0],
                                  cflt.stride[0], cflt.padding[0]);
        dst[3] = infer_conv_shape(src[3], cflt.dilated_spatial[1],
                                  cflt.stride[1], cflt.padding[1]);
        dst[4] = 32;
    } else if (param().format == Param::Format::NCHW88 ||
               param().format == Param::Format::NCHW88_WINOGRAD) {
        megdnn_assert(src.ndim == 5 || (src.ndim == 4 && src[1] <= 8),
                      "invalid src ndim for NCHW88, expected=5 or 4, got=%zu",
                      src.ndim);
        dst.ndim = 5;
        dst[0] = src[0];
        auto oc = cflt.ocpg * cflt.group;
        megdnn_assert(oc % 8 == 0);
        dst[1] = oc / 8;
        dst[2] = infer_conv_shape(src[2], cflt.dilated_spatial[0],
                                  cflt.stride[0], cflt.padding[0]);
        dst[3] = infer_conv_shape(src[3], cflt.dilated_spatial[1],
                                  cflt.stride[1], cflt.padding[1]);
        dst[4] = 8;
        if (cflt.group == 1) {
            megdnn_assert(cflt.icpg * cflt.group == src[1] * 8 ||
                                  (cflt.icpg * cflt.group == src[1]),
                          "%s icpg=%u group=%u", errmsg().c_str(), cflt.icpg,
                          cflt.group);
        }

    } else if (param().format == Param::Format::NCHW44 ||
               param().format == Param::Format::NCHW44_DOT ||
               param().format == Param::Format::NCHW44_WINOGRAD) {
        megdnn_assert(src.ndim == 5 || (src.ndim == 4 && src[1] <= 4),
                      "invalid src ndim for NCHW44, expected=5 or 4, got=%zu",
                      src.ndim);
        dst.ndim = 5;
        dst[0] = src[0];
        auto oc = cflt.ocpg * cflt.group;
        megdnn_assert(oc % 4 == 0);
        dst[1] = oc / 4;
        dst[2] = infer_conv_shape(src[2], cflt.dilated_spatial[0],
                                  cflt.stride[0], cflt.padding[0]);
        dst[3] = infer_conv_shape(src[3], cflt.dilated_spatial[1],
                                  cflt.stride[1], cflt.padding[1]);
        dst[4] = 4;
        if (cflt.group == 1) {
            megdnn_assert(cflt.icpg * cflt.group == src[1] * 4 ||
                                  (cflt.icpg * cflt.group == src[1]),
                          "%s icpg=%u group=%u", errmsg().c_str(), cflt.icpg,
                          cflt.group);
        }
    } else if (param().format == Param::Format::CHWN4) {
        megdnn_assert(src.ndim == 5,
                      "invalid src ndim for CHWN4, expected=5, got=%zu",
                      src.ndim);
        megdnn_assert(cflt.icpg * cflt.group == src[0] * 4,
                      "%s icpg=%u group=%u", errmsg().c_str(), cflt.icpg,
                      cflt.group);
        dst.ndim = src.ndim;
        dst[3] = src[3];
        auto oc = cflt.ocpg * cflt.group;
        megdnn_assert(oc % 4 == 0);
        dst[0] = oc / 4;
        dst[1] = infer_conv_shape(src[1], cflt.dilated_spatial[0],
                                  cflt.stride[0], cflt.padding[0]);
        dst[2] = infer_conv_shape(src[2], cflt.dilated_spatial[1],
                                  cflt.stride[1], cflt.padding[1]);
        dst[4] = 4;
    } else if (param().format == Param::Format::NCHW4_NCHW) {
        megdnn_assert(src.ndim == 5,
                      "invalid src ndim for NCHW4_NCHW, expected=5, got=%zu",
                      src.ndim);
        megdnn_assert(cflt.icpg * cflt.group == src[1] * 4,
                      "%s icpg=%u group=%u", errmsg().c_str(), cflt.icpg,
                      cflt.group);
        dst.ndim = 4;
        dst[0] = src[0];
        auto oc = cflt.ocpg * cflt.group;
        dst[1] = oc;
        dst[2] = infer_conv_shape(src[2], cflt.dilated_spatial[0],
                                  cflt.stride[0], cflt.padding[0]);
        dst[3] = infer_conv_shape(src[3], cflt.dilated_spatial[1],
                                  cflt.stride[1], cflt.padding[1]);
    } else if (param().format == Param::Format::NCHW4_NCHW32) {
        megdnn_assert(src.ndim == 5,
                      "invalid src ndim for NCHW4_NCHW32, expected=5, got=%zu",
                      src.ndim);
        megdnn_assert(cflt.icpg * cflt.group == src[1] * 4,
                      "%s icpg=%u group=%u", errmsg().c_str(), cflt.icpg,
                      cflt.group);
        dst.ndim = src.ndim;
        dst[0] = src[0];
        auto oc = cflt.ocpg * cflt.group;
        megdnn_assert(oc % 32 == 0);
        dst[1] = oc / 32;
        dst[2] = infer_conv_shape(src[2], cflt.dilated_spatial[0],
                                  cflt.stride[0], cflt.padding[0]);
        dst[3] = infer_conv_shape(src[3], cflt.dilated_spatial[1],
                                  cflt.stride[1], cflt.padding[1]);
        dst[4] = 32;
    } else if (param().format == Param::Format::NCHW32_NCHW4) {
        megdnn_assert(src.ndim == 5,
                      "invalid src ndim for NCHW32_NCHW4, expected=5, got=%zu",
                      src.ndim);
        megdnn_assert(cflt.icpg * cflt.group == src[1] * 32,
                      "%s icpg=%u group=%u", errmsg().c_str(), cflt.icpg,
                      cflt.group);
        dst.ndim = src.ndim;
        dst[0] = src[0];
        auto oc = cflt.ocpg * cflt.group;
        megdnn_assert(oc % 4 == 0);
        dst[1] = oc / 4;
        dst[2] = infer_conv_shape(src[2], cflt.dilated_spatial[0],
                                  cflt.stride[0], cflt.padding[0]);
        dst[3] = infer_conv_shape(src[3], cflt.dilated_spatial[1],
                                  cflt.stride[1], cflt.padding[1]);
        dst[4] = 4;
    } else {
        megdnn_assert(param().format == Param::Format::NHWCD4);
        megdnn_assert(src.ndim == 5,
                      "invalid src ndim for NHWCD4, expected=5, got=%zu",
                      src.ndim);
        megdnn_assert(cflt.icpg * cflt.group == src[2] * 4,
                      "%s icpg=%u group=%u", errmsg().c_str(), cflt.icpg,
                      cflt.group);
        dst.ndim = src.ndim;
        dst[0] = src[0];
        auto oc = cflt.ocpg * cflt.group;
        megdnn_assert(oc % 4 == 0);
        dst[2] = oc / 4;
        dst[1] = infer_conv_shape(src[1], cflt.dilated_spatial[0],
                                  cflt.stride[0], cflt.padding[0]);
        dst[3] = infer_conv_shape(src[3], cflt.dilated_spatial[1],
                                  cflt.stride[1], cflt.padding[1]);
        megdnn_assert(src[4] == 4);
        dst[4] = 4;
    }
    dst.format = src.format;
    dst.init_contiguous_stride();
    return cflt;
}

/**
 * \warning: An explicit specialization shall be declared in a namespace
 * enclosing the specialized template. An explicit specialization whose
 * declarator-id is not qualified shall be declared in the nearest enclosing
 * namespace of the template, or, if the namespace is inline (7.3.1), any
 * namespace from its enclosing namespace set.
 * refer to:
 * https://stackoverflow.com/questions/25594644/warning-specialization-of-template-in-different-namespace
 */
template <>
ConvolutionBase<param::Convolution>::CanonizedFilterMeta
ConvolutionBase<param::Convolution>::check_layout_fwd(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst) const {
    TensorLayout dst_expected;
    dst_expected.dtype = dst.dtype;

    auto ret = deduce_layout_fwd(src, filter, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
    return ret;
}

template <>
ConvolutionBase<param::ConvBias>::CanonizedFilterMeta
ConvolutionBase<param::ConvBias>::check_layout_fwd(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst) const {
    TensorLayout dst_expected;
    dst_expected.dtype = dst.dtype;

    auto ret = deduce_layout_fwd(src, filter, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
    return ret;
}

template <>
ConvolutionBase<param::BatchConvBias>::CanonizedFilterMeta
ConvolutionBase<param::BatchConvBias>::check_layout_fwd(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst) const {
    TensorLayout dst_expected;
    dst_expected.dtype = dst.dtype;

    auto ret = deduce_layout_fwd(src, filter, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
    return ret;
}

void ConvolutionForward::deduce_dtype(DType src, DType filter, DType& dst) {
    check_or_deduce_dtype_fwd(src, filter, dst);
}

void ConvolutionForward::deduce_layout(const TensorLayout& src,
                                       const TensorLayout& filter,
                                       TensorLayout& dst) {
    deduce_layout_fwd(src, filter, dst);
}

ConvolutionForward::CanonizedFilterMeta ConvolutionForward::check_exec(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst, size_t workspace_in_bytes,
        const PreprocessedFilter* preprocessed_filter) {
    auto ret = check_layout_fwd(src, filter, dst);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, filter, dst, preprocessed_filter);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}

ConvolutionBackwardData::CanonizedFilterMeta
ConvolutionBackwardData::check_exec(const TensorLayout& filter,
                                    const TensorLayout& diff,
                                    const TensorLayout& grad,
                                    size_t workspace_in_bytes) {
    auto grad_fwd = grad;
    auto filter_fwd = filter;
    auto diff_fwd = diff;

    std::swap(grad_fwd.dtype, diff_fwd.dtype);

    grad_fwd.init_contiguous_stride();
    diff_fwd.init_contiguous_stride();
    auto ret = check_layout_fwd(grad_fwd, filter_fwd, diff_fwd);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(filter, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}

void ConvolutionBackwardData::deduce_dtype(DType filter, DType diff,
                                           DType& grad) {
    SmallVector<DType> supported_dst_dtype;
    if (filter.category() == diff.category() &&
        filter.category() == DTypeCategory::FLOAT) {
        supported_dst_dtype.push_back(filter);
    } else if (filter.enumv() == DTypeEnum::Int8 && diff == filter) {
        supported_dst_dtype.push_back(dtype::Int32());
    } else if ((filter.enumv() == DTypeEnum::QuantizedS8 &&
                diff.enumv() == DTypeEnum::QuantizedS8) ||
               (filter.enumv() == DTypeEnum::Quantized8Asymm &&
                diff.enumv() == DTypeEnum::Quantized8Asymm)) {
        supported_dst_dtype.push_back(
                dtype::QuantizedS32(mul_scale(filter, diff)));
        if (grad.valid() && grad.enumv() == diff.enumv()) {
            supported_dst_dtype.push_back(grad);
        }
    } else {
        megdnn_throw(ssprintf("unsupported input / diff DType: %s x %s",
                              filter.name(), diff.name()));
    }
    if (!grad.valid()) {
        grad = supported_dst_dtype.at(0);
    } else {
        megdnn_assert(vec_contains(supported_dst_dtype, grad),
                      "unsupported ConvBwd(%s, %s) -> %s", filter.name(),
                      diff.name(), grad.name());
    }
    megdnn_assert(param().compute_mode != Param::ComputeMode::FLOAT32
#if !MEGDNN_DISABLE_FLOAT16
                          || filter.enumv() == DTypeEnum::Float16
                          || filter.enumv() == DTypeEnum::BFloat16
#endif
                          ,
                  "ComputeMode::FLOAT32 is only available for Float16/BFloat16 "
                  "input / output.");
}

void ConvolutionBackwardData::deduce_layout(const TensorLayout& filter,
                                            const TensorLayout& diff,
                                            TensorLayout& grad) {
    auto errmsg = [&]() { return get_errmsg(filter, diff, grad, param()); };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(filter);
    megdnn_assert_contiguous(diff);
    megdnn_assert(filter.ndim == 4_z || filter.ndim == 5_z, "%s",
                  errmsg().c_str());
    megdnn_assert(diff.ndim == 4_z || diff.ndim == 5_z, "%s", errmsg().c_str());

    deduce_dtype(filter.dtype, diff.dtype, grad.dtype);

    auto cflt = make_canonized_filter_meta(diff.ndim, filter);

    auto deduce = [&errmsg](size_t out, size_t filter, size_t stride,
                            size_t pad) {
        MEGDNN_MARK_USED_VAR(errmsg);
        auto i = (out - 1) * stride + filter;
        megdnn_assert(i > pad * 2, "%s", errmsg().c_str());
        return i - pad * 2;
    };

    if (param().format == Param::Format::NCHW ||
        param().format == Param::Format::NHWC) {
        size_t src_or_dst_c_pos = 0;
        size_t src_or_dst_spatial_start = 0;
        if (param().format == Param::Format::NCHW) {
            src_or_dst_c_pos = 1;
            src_or_dst_spatial_start = 2;
        } else {
            megdnn_assert(param().format == Param::Format::NHWC,
                          "invalid conv format");
            src_or_dst_c_pos = 3;
            src_or_dst_spatial_start = 1;
        }
        megdnn_assert(cflt.ocpg * cflt.group == diff[src_or_dst_c_pos], "%s",
                      errmsg().c_str());
        grad.ndim = diff.ndim;
        grad[0] = diff[0];
        grad[src_or_dst_c_pos] = cflt.icpg * cflt.group;
        for (size_t i = 0; i < cflt.spatial_ndim; ++i) {
            grad[i + src_or_dst_spatial_start] = deduce(
                    diff[i + src_or_dst_spatial_start], cflt.dilated_spatial[i],
                    cflt.stride[i], cflt.padding[i]);
        }
    } else {
        megdnn_assert(param().format == Param::Format::NHWCD4);
        megdnn_assert(diff.ndim == 5,
                      "valid diff ndim for NHWCD4, expected=5, got=%zu",
                      diff.ndim);
        megdnn_assert(cflt.ocpg * cflt.group == diff[2] * 4, "%s",
                      errmsg().c_str());
        grad.ndim = diff.ndim;
        grad[0] = diff[0];
        auto ic = cflt.icpg * cflt.group;
        megdnn_assert(ic % 4 == 0);
        grad[2] = ic / 4;
        grad[1] = deduce(diff[1], cflt.dilated_spatial[0], cflt.stride[0],
                         cflt.padding[0]);
        grad[3] = deduce(diff[3], cflt.dilated_spatial[1], cflt.stride[1],
                         cflt.padding[1]);
        megdnn_assert(diff[4] == 4);
        grad[4] = 4;
    }
    grad.format = diff.format;
    grad.init_contiguous_stride();
}

ConvolutionBackwardFilter::CanonizedFilterMeta
ConvolutionBackwardFilter::check_exec(const TensorLayout& src,
                                      const TensorLayout& diff,
                                      const TensorLayout& grad,
                                      size_t workspace_in_bytes) {
    megdnn_assert(src.dtype.category() == DTypeCategory::FLOAT &&
                          diff.dtype.category() == DTypeCategory::FLOAT &&
                          grad.dtype.category() == DTypeCategory::FLOAT,
                  "only float type is supported for conv backward filter");
    auto ret = check_layout_fwd(src, grad, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
