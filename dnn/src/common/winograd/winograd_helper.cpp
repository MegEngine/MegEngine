/**
 * \file dnn/src/common/winograd/winograd_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/winograd/winograd_helper.h"
#include "src/common/winograd/winograd_generator.h"
#include "src/naive/matrix_mul/matrix_mul_helper.h"

using namespace megdnn;
namespace {
template <typename ctype, typename otype, typename enable = void>
struct Getter {
    Getter(DType){};
    otype operator()(ctype item) { return item; }
};

template <typename ctype, typename otype>
struct Getter<ctype, otype,
              typename std::enable_if_t<std::is_same<ctype, uint8_t>::value>> {
    otype zp;
    Getter(DType dtype) {
        zp = dtype.param<dtype::Quantized8Asymm>().zero_point;
    }
    otype operator()(ctype item) { return static_cast<otype>(item) - zp; }
};

template <typename ctype, typename otype, typename enable = void>
struct OutputGetter {
    OutputGetter(DType){};
    otype operator()(float item) { return static_cast<otype>(item); }
};

template <typename ctype, typename otype>
struct OutputGetter<
        ctype, otype,
        typename std::enable_if_t<std::is_same<otype, int8_t>::value>> {
    DType dtype;
    OutputGetter(DType dtype) : dtype{dtype} {}
    otype operator()(float item) {
        return dtype.param<dtype::QuantizedS8>().quantize(item).as_int8();
    }
};

template <typename ctype, typename otype>
struct OutputGetter<
        ctype, otype,
        typename std::enable_if_t<std::is_same<otype, uint8_t>::value>> {
    DType dtype;
    OutputGetter(DType dtype) : dtype{dtype} {}
    otype operator()(float item) {
        return dtype.param<dtype::Quantized8Asymm>().quantize(item).as_uint8();
    }
};
}  // namespace

namespace megdnn {
namespace winograd {

constexpr size_t layout_pack_size(param::ConvBias::Format layout) {
    switch (layout) {
        case param::ConvBias::Format::NHWCD4:
            return 4;
        case param::ConvBias::Format::NCHW44:
        case param::ConvBias::Format::NCHW4:
            return 4;
        case param::ConvBias::Format::NCHW32:
            return 32;
        case param::ConvBias::Format::NCHW88:
        case param::ConvBias::Format::NCHW8:
            return 8;
        default:
            return 1;
    }
}

template <param::ConvBias::Format layout, param::MatrixMul::Format format>
struct FilterVisitor {
    size_t IC, OC;
    FilterVisitor(size_t OC, size_t IC) : IC(IC), OC(OC) {}
    size_t get(size_t r, size_t oc, size_t ic, size_t h, size_t w) {
        constexpr size_t input_pack_size = layout_pack_size(layout);
        size_t ocb_layout = oc / input_pack_size;
        size_t oc_layout = oc % input_pack_size;
        size_t icb_layout = ic / input_pack_size;
        size_t ic_layout = ic % input_pack_size;

        return (ocb_layout * (IC / input_pack_size) + icb_layout) * r * r *
                       input_pack_size * input_pack_size +
               ic_layout * input_pack_size + oc_layout +
               (h * r + w) * input_pack_size * input_pack_size;
    }

    size_t put(size_t alpha, size_t oc, size_t ic, size_t h, size_t w) {
        if (format == param::MatrixMul::Format::DEFAULT) {
            return (h * alpha + w) * OC * IC + ic * OC + oc;
        }
        size_t matmul_pack_size = MatrixMulForward::pack_size(format);
        size_t ocb = oc / matmul_pack_size;
        size_t oc_pack = oc % matmul_pack_size;
        size_t icb = ic / matmul_pack_size;
        size_t ic_pack = ic % matmul_pack_size;

        size_t OCB = OC / matmul_pack_size;
        size_t ICB = IC / matmul_pack_size;

        return (h * alpha + w) * OCB * ICB * matmul_pack_size *
                       matmul_pack_size +
               ocb * ICB * matmul_pack_size * matmul_pack_size +
               icb * matmul_pack_size * matmul_pack_size +
               ic_pack * matmul_pack_size + oc_pack;
    }
};

template <param::ConvBias::Format layout, param::MatrixMul::Format format>
struct InputVisitor {
    size_t IC;
    InputVisitor(size_t IC) : IC(IC) {}

    size_t get(size_t /*alpha*/, size_t ic, size_t IH, size_t IW, size_t ih,
               size_t iw) {
        constexpr size_t input_pack_size = layout_pack_size(layout);
        size_t icb_layout = ic / input_pack_size;
        size_t ic_layout = ic % input_pack_size;

        return (icb_layout * IH * IW + ih * IW + iw) * input_pack_size +
               ic_layout;
    }

    size_t put(size_t alpha, size_t ic, size_t nr_units_in_tile,
               size_t unit_idx, size_t h, size_t w) {
        if (format == param::MatrixMul::Format::DEFAULT) {
            return (h * alpha + w) * nr_units_in_tile * IC + unit_idx * IC + ic;
        }
        size_t matmul_pack_size = MatrixMulForward::pack_size(format);
        size_t icb = ic / matmul_pack_size;
        size_t ic_pack = ic % matmul_pack_size;
        size_t ICB = IC / matmul_pack_size;

        return (h * alpha + w) * ICB * nr_units_in_tile * matmul_pack_size +
               icb * nr_units_in_tile * matmul_pack_size +
               unit_idx * matmul_pack_size + ic_pack;
    }
};

template <param::ConvBias::Format layout, param::MatrixMul::Format format>
struct OutputVisitor {
    size_t OC;
    OutputVisitor(size_t OC) : OC(OC) {}

    size_t get(size_t alpha, size_t oc_index, size_t oc,
               size_t nr_units_in_tile, size_t unit_idx, size_t h, size_t w) {
        if (format == param::MatrixMul::Format::DEFAULT) {
            return (h * alpha + w) * nr_units_in_tile * OC + unit_idx * OC +
                   oc_index;
        }
        size_t matmul_pack_size = MatrixMulForward::pack_size(format);
        size_t ocb = oc_index / matmul_pack_size;
        size_t oc_pack = oc % matmul_pack_size;
        size_t OCB = OC / matmul_pack_size;

        return (h * alpha + w) * OCB * nr_units_in_tile * matmul_pack_size +
               ocb * nr_units_in_tile * matmul_pack_size +
               unit_idx * matmul_pack_size + oc_pack;
    }

    size_t put(size_t oc, size_t OH, size_t OW, size_t oh, size_t ow) {
        constexpr size_t input_pack_size = layout_pack_size(layout);
        size_t oc_layout = oc % input_pack_size;

        return (oc / input_pack_size * OH * OW + oh * OW + ow) *
                       input_pack_size +
               oc_layout;
    }
};

template <typename ctype, typename dst_type, typename input_filter_compute_type,
          typename output_compute_type, param::ConvBias::Format layout,
          param::MatrixMul::Format format>
void StrategyHelper<
        ctype, dst_type, input_filter_compute_type, output_compute_type, layout,
        format>::filter(const ctype* filter,
                        input_filter_compute_type* filter_transform_buf,
                        input_filter_compute_type* transform_mid_buf, size_t OC,
                        size_t IC, size_t oc_start, size_t oc_end, size_t m,
                        size_t r, const std::vector<float>& interp_points,
                        DType dtype, float rescale) {
    size_t alpha = m + r - 1;
    WinogradCoeff<input_filter_compute_type> winograd_coeff(m, r,
                                                            interp_points);
    input_filter_compute_type* mid_buf1 = transform_mid_buf;
    input_filter_compute_type* mid_buf2 = transform_mid_buf + alpha * alpha;
    Getter<ctype, input_filter_compute_type> getter(dtype);
    FilterVisitor<layout, format> filter_visitor(OC, IC);

    for (size_t oc = oc_start; oc < oc_end; oc++) {
        rep(ic, IC) {
            rep(i, r) rep(j, r) {
                mid_buf1[i * r + j] =
                        getter(filter[filter_visitor.get(r, oc, ic, i, j)]);
            }

            /* tmp = Matmul(G, src) */
            megdnn::naive::run_matrix_mul_tpl<input_filter_compute_type,
                                              input_filter_compute_type, false,
                                              false>(
                    winograd_coeff.G(rescale).data(), mid_buf1, mid_buf2, alpha,
                    r, r, r, r, r, dtype, dtype);
            /* dst = Matmul(tmp, G^T) */
            megdnn::naive::run_matrix_mul_tpl<input_filter_compute_type,
                                              input_filter_compute_type, false,
                                              true>(
                    mid_buf2, winograd_coeff.G(rescale).data(), mid_buf1, alpha,
                    alpha, r, r, r, alpha, dtype, dtype);

            rep(i, alpha) rep(j, alpha) {
                filter_transform_buf[filter_visitor.put(alpha, oc, ic, i, j)] =
                        mid_buf1[i * alpha + j];
            }
        }
    }
}

template <typename ctype, typename dst_type, typename input_filter_compute_type,
          typename output_compute_type, param::ConvBias::Format layout,
          param::MatrixMul::Format format>
void StrategyHelper<
        ctype, dst_type, input_filter_compute_type, output_compute_type, layout,
        format>::input(const ctype* input,
                       input_filter_compute_type* input_transform_buf,
                       input_filter_compute_type* transform_mid_buf,
                       int ih_start, int iw_start, size_t IH, size_t IW,
                       size_t IC, size_t ic, size_t unit_idx, size_t nr_units_in_tile,
                       size_t m, size_t r,
                       const std::vector<float>& interp_points, DType dtype,
                       float rescale) {
    size_t alpha = m + r - 1;
    WinogradCoeff<input_filter_compute_type> winograd_coeff(m, r,
                                                            interp_points);
    input_filter_compute_type* mid_buf1 = transform_mid_buf;
    input_filter_compute_type* mid_buf2 = transform_mid_buf + alpha * alpha;
    Getter<ctype, input_filter_compute_type> getter(dtype);
    InputVisitor<layout, format> intput_visitor(IC);

    memset(mid_buf1, 0, alpha * alpha * sizeof(input_filter_compute_type));
    rep(i, alpha) rep(j, alpha) {
        int ih = ih_start + i;
        int iw = iw_start + j;
        if (ih >= 0 && ih < (int)IH && iw >= 0 && iw < (int)IW) {
            mid_buf1[i * alpha + j] = getter(
                    input[intput_visitor.get(alpha, ic, IH, IW, ih, iw)]);
        }
    }

    megdnn::naive::run_matrix_mul_tpl<input_filter_compute_type,
                                        input_filter_compute_type, true,
                                        false>(
            winograd_coeff.B(rescale).data(), mid_buf1, mid_buf2, alpha,
            alpha, alpha, alpha, alpha, alpha, dtype, dtype);
    megdnn::naive::run_matrix_mul_tpl<input_filter_compute_type,
                                        input_filter_compute_type, false,
                                        false>(
            mid_buf2, winograd_coeff.B(rescale).data(), mid_buf1, alpha,
            alpha, alpha, alpha, alpha, alpha, dtype, dtype);

    rep(i, alpha) rep(j, alpha) {
        input_transform_buf[intput_visitor.put(alpha, ic, nr_units_in_tile,
                                                unit_idx, i, j)] =
                mid_buf1[i * alpha + j];
    }
}

template <typename ctype, typename dst_type, typename input_filter_compute_type,
          typename output_compute_type, param::ConvBias::Format layout,
          param::MatrixMul::Format format>
void StrategyHelper<
        ctype, dst_type, input_filter_compute_type, output_compute_type, layout,
        format>::output(const output_compute_type* output_transform_buf,
                        const output_compute_type* bias, dst_type* output,
                        output_compute_type* transform_mid_buf, BiasMode bmode,
                        NonlineMode nonline_mode, size_t oh_start,
                        size_t ow_start, size_t OH, size_t OW, size_t OC, size_t oc_start,
                        size_t oc_index, size_t unit_idx, size_t nr_units_in_tile,
                        size_t m, size_t r,
                        const std::vector<float>& interp_points, DType dtype,
                        float input_filter_scale, float input_filter_rescale,
                        float rescale) {
    size_t alpha = m + r - 1;
    winograd::WinogradCoeff<output_compute_type> winograd_coeff(m, r,
                                                                interp_points);
    output_compute_type* mid_buf1 = transform_mid_buf;
    output_compute_type* mid_buf2 = transform_mid_buf + alpha * alpha;
    OutputGetter<output_compute_type, dst_type> getter(dtype);
    OutputVisitor<layout, format> output_visitor(OC);

    size_t oc = oc_start + oc_index;

    /* gather */
    rep(i, alpha) rep(j, alpha) {
        mid_buf1[i * alpha + j] = output_transform_buf[output_visitor.get(
                alpha, oc_index, oc, nr_units_in_tile, unit_idx, i,
                j)];
    }
    /* A[alpha*m] M[alpha*alpha] */
    megdnn::naive::run_matrix_mul_tpl<output_compute_type,
                                        output_compute_type, true, false>(
            winograd_coeff.A(rescale).data(), mid_buf1, mid_buf2, m, alpha,
            alpha, m, alpha, alpha, dtype, dtype);
    megdnn::naive::run_matrix_mul_tpl<output_compute_type,
                                        output_compute_type, false, false>(
            mid_buf2, winograd_coeff.A(rescale).data(), mid_buf1, m, m,
            alpha, alpha, m, m, dtype, dtype);

    rep(i, m) rep(j, m) {
        auto oh = oh_start + i;
        auto ow = ow_start + j;
        if (oh < OH && ow < OW) {
            float val = mid_buf1[i * m + j];
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                val += bias[oc] * input_filter_rescale *
                        input_filter_rescale;
            } else if (bmode == BiasMode::BIAS) {
                val += bias[output_visitor.put(oc, OH, OW, oh, ow)] *
                        input_filter_rescale * input_filter_rescale;
            }
            val = val * input_filter_scale /
                    (input_filter_rescale * input_filter_rescale * rescale *
                    rescale);
            if (nonline_mode == NonlineMode::RELU) {
                val = val > 0 ? val : 0;
            } else if (nonline_mode == NonlineMode::SIGMOID) {
                val = 1.f / (expf(-val) + 1.f);
            } else if (nonline_mode == NonlineMode::H_SWISH) {
                val = val * std::min(std::max(val + 3, 0.f), 6.f) / 6.f;
            } else {
                megdnn_assert(nonline_mode == NonlineMode::IDENTITY);
            }
            output[output_visitor.put(oc, OH, OW, oh, ow)] = getter(val);
        }
    }
};

#define INST(_ctype, _dst_type, _input_filter_compute_type,   \
             _output_compute_type)                            \
    template class StrategyHelper<_ctype, _dst_type,          \
                                  _input_filter_compute_type, \
                                  _output_compute_type>;

INST(float, float, float, float)
MEGDNN_INC_FLOAT16(INST(dt_float16, dt_float16, dt_float16, dt_float16))
INST(int8_t, int8_t, int16_t, int)
INST(uint8_t, uint8_t, int16_t, int)
#undef INST

#define INST(_ctype, _dst_type, _input_filter_compute_type, \
             _output_compute_type, layout)                  \
    template class StrategyHelper<                          \
            _ctype, _dst_type, _input_filter_compute_type,  \
            _output_compute_type, layout, param::MatrixMul::Format::MK4>;
INST(float, float, float, float, param::ConvBias::Format::NCHW)
INST(float, float, float, float, param::ConvBias::Format::NCHW44)
INST(int8_t, int8_t, float, float, param::ConvBias::Format::NCHW44)
#undef INST

#define INST(_ctype, _dst_type, _input_filter_compute_type, \
             _output_compute_type, layout)                  \
    template class StrategyHelper<                          \
            _ctype, _dst_type, _input_filter_compute_type,  \
            _output_compute_type, layout, param::MatrixMul::Format::MK8>;
INST(int8_t, int8_t, int16_t, int, param::ConvBias::Format::NCHW)
INST(int8_t, int8_t, int16_t, int, param::ConvBias::Format::NCHW44)
INST(float, float, float, float, param::ConvBias::Format::NCHW88)
MEGDNN_INC_FLOAT16(INST(dt_float16, dt_float16, dt_float16, dt_float16,
                        param::ConvBias::Format::NCHW))
#undef INST
}  // namespace winograd
}  // namespace megdnn

// vim: syntax=cpp.doxygen
