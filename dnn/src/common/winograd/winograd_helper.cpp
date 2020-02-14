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

template <typename ctype, typename dst_type, typename input_filter_compute_type,
          typename output_compute_type>
class StrategyHelper<ctype, dst_type, input_filter_compute_type,
                     output_compute_type, param::MatrixMul::Format::DEFAULT> {
public:
    static void filter(const ctype* filter,
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
        for (size_t oc = oc_start; oc < oc_end; oc++) {
            rep(ic, IC) {
                const ctype* filter_ptr = filter + (oc * IC + ic) * r * r;
                rep(i, r) rep(j, r) {
                    mid_buf1[i * r + j] = getter(filter_ptr[i * r + j]);
                }

                /* tmp = Matmul(G, src) */
                megdnn::naive::run_matrix_mul_tpl<input_filter_compute_type,
                                                  input_filter_compute_type,
                                                  false, false>(
                        winograd_coeff.G(rescale).data(), mid_buf1, mid_buf2,
                        alpha, r, r, r, r, r, dtype, dtype);
                /* dst = Matmul(tmp, G^T) */
                megdnn::naive::run_matrix_mul_tpl<input_filter_compute_type,
                                                  input_filter_compute_type,
                                                  false, true>(
                        mid_buf2, winograd_coeff.G(rescale).data(), mid_buf1,
                        alpha, alpha, r, r, r, alpha, dtype, dtype);

                rep(i, alpha) rep(j, alpha) {
                    filter_transform_buf[(i * alpha + j) * OC * IC + ic * OC +
                                         oc] = mid_buf1[i * alpha + j];
                }
            }
        }
    }

    static void input(const ctype* input,
                      input_filter_compute_type* input_transform_buf,
                      input_filter_compute_type* transform_mid_buf,
                      int ih_start, int iw_start, size_t IH, size_t IW,
                      size_t IC, size_t unit_idx, size_t nr_units_in_tile,
                      size_t m, size_t r,
                      const std::vector<float>& interp_points, DType dtype,
                      float rescale) {
        size_t alpha = m + r - 1;
        Getter<ctype, input_filter_compute_type> getter(dtype);
        WinogradCoeff<input_filter_compute_type> winograd_coeff(m, r,
                                                                interp_points);
        rep(ic, IC) {
            input_filter_compute_type* mid_buf1 = transform_mid_buf;
            input_filter_compute_type* mid_buf2 =
                    transform_mid_buf + alpha * alpha;

            memset(mid_buf1, 0,
                   alpha * alpha * sizeof(input_filter_compute_type));
            rep(i, alpha) rep(j, alpha) {
                int ih = ih_start + i;
                int iw = iw_start + j;
                if (ih >= 0 && ih < (int)IH && iw >= 0 && iw < (int)IW) {
                    mid_buf1[i * alpha + j] =
                            getter(input[ic * IH * IW + ih * IW + iw]);
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
                input_transform_buf[(i * alpha + j) * nr_units_in_tile * IC +
                                    unit_idx * IC + ic] =
                        mid_buf1[i * alpha + j];
            }
        }
    }

    static void output(const output_compute_type* output_transform_buf,
                       const output_compute_type* bias, dst_type* output,
                       output_compute_type* transform_mid_buf, BiasMode bmode,
                       NonlineMode nonline_mode, size_t oh_start,
                       size_t ow_start, size_t OH, size_t OW, size_t oc_start,
                       size_t oc_end, size_t unit_idx, size_t nr_units_in_tile,
                       size_t m, size_t r,
                       const std::vector<float>& interp_points, DType dtype,
                       float input_filter_scale, float input_filter_rescale,
                       float rescale) {
        size_t alpha = m + r - 1;
        size_t OC = oc_end - oc_start;

        OutputGetter<output_compute_type, dst_type> getter(dtype);
        winograd::WinogradCoeff<output_compute_type> winograd_coeff(
                m, r, interp_points);
        for (size_t oc = oc_start; oc < oc_end; oc++) {
            size_t oc_index = oc - oc_start;
            output_compute_type* mid_buf1 = transform_mid_buf;
            output_compute_type* mid_buf2 = transform_mid_buf + alpha * alpha;

            // gather
            rep(i, alpha) rep(j, alpha) {
                mid_buf1[i * alpha + j] =
                        output_transform_buf[(i * alpha + j) *
                                                     nr_units_in_tile * OC +
                                             unit_idx * OC + oc_index];
            }
            /* A[alpha*m] M[alpha*alpha] */
            megdnn::naive::run_matrix_mul_tpl<output_compute_type,
                                              output_compute_type, true, false>(
                    winograd_coeff.A(rescale).data(), mid_buf1, mid_buf2, m,
                    alpha, alpha, m, alpha, alpha, dtype, dtype);
            megdnn::naive::run_matrix_mul_tpl<
                    output_compute_type, output_compute_type, false, false>(
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
                        val += bias[oc * OH * OW + oh * OW + ow] *
                               input_filter_rescale * input_filter_rescale;
                    }
                    val = val * input_filter_scale /
                          (input_filter_rescale * input_filter_rescale *
                           rescale * rescale);
                    if (nonline_mode == NonlineMode::RELU) {
                        val = val > 0 ? val : 0;
                    } else if (nonline_mode == NonlineMode::SIGMOID) {
                        val = 1.f / (expf(-val) + 1.f);
                    } else if (nonline_mode == NonlineMode::H_SWISH) {
                        val = val * std::min(std::max(val + 3, 0.f), 6.f) / 6.f;
                    } else {
                        megdnn_assert(nonline_mode == NonlineMode::IDENTITY);
                    }

                    output[oc * OH * OW + oh * OW + ow] = getter(val);
                }
            }
        }
    }
};

template <typename ctype, typename dst_type, typename input_filter_compute_type,
          typename output_compute_type, param::MatrixMul::Format format>
class StrategyHelper<
        ctype, dst_type, input_filter_compute_type, output_compute_type, format,
        std::enable_if_t<format == param::MatrixMul::Format::MK4 ||
                         format == param::MatrixMul::Format::MK8>> {
public:
    static void filter(const ctype* filter,
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
        size_t OCB = OC / pack_size;
        size_t ICB = IC / pack_size;
        for (size_t oc = oc_start; oc < oc_end; oc++) {
            rep(ic, IC) {
                const ctype* filter_ptr = filter + (oc * IC + ic) * r * r;
                rep(i, r) rep(j, r) {
                    mid_buf1[i * r + j] = getter(filter_ptr[i * r + j]);
                }

                /* tmp = Matmul(G, src) */
                megdnn::naive::run_matrix_mul_tpl<input_filter_compute_type,
                                                  input_filter_compute_type,
                                                  false, false>(
                        winograd_coeff.G(rescale).data(), mid_buf1, mid_buf2,
                        alpha, r, r, r, r, r, dtype, dtype);
                /* dst = Matmul(tmp, G^T) */
                megdnn::naive::run_matrix_mul_tpl<input_filter_compute_type,
                                                  input_filter_compute_type,
                                                  false, true>(
                        mid_buf2, winograd_coeff.G(rescale).data(), mid_buf1,
                        alpha, alpha, r, r, r, alpha, dtype, dtype);

                size_t ocb = oc / pack_size;
                size_t oc_pack = oc % pack_size;
                size_t icb = ic / pack_size;
                size_t ic_pack = ic % pack_size;
                rep(i, alpha) rep(j, alpha) {
                    filter_transform_buf[(i * alpha + j) * OCB * ICB *
                                                 pack_size * pack_size +
                                         ocb * ICB * pack_size * pack_size +
                                         icb * pack_size * pack_size +
                                         ic_pack * pack_size + oc_pack] =
                            mid_buf1[i * alpha + j];
                }
            }
        }
    }

    static void input(const ctype* input,
                      input_filter_compute_type* input_transform_buf,
                      input_filter_compute_type* transform_mid_buf,
                      int ih_start, int iw_start, size_t IH, size_t IW,
                      size_t IC, size_t unit_idx, size_t nr_units_in_tile,
                      size_t m, size_t r,
                      const std::vector<float>& interp_points, DType dtype,
                      float rescale) {
        size_t alpha = m + r - 1;
        Getter<ctype, input_filter_compute_type> getter(dtype);
        WinogradCoeff<input_filter_compute_type> winograd_coeff(m, r,
                                                                interp_points);
        size_t ICB = IC / pack_size;
        rep(ic, IC) {
            input_filter_compute_type* mid_buf1 = transform_mid_buf;
            input_filter_compute_type* mid_buf2 =
                    transform_mid_buf + alpha * alpha;

            memset(mid_buf1, 0,
                   alpha * alpha * sizeof(input_filter_compute_type));
            rep(i, alpha) rep(j, alpha) {
                int ih = ih_start + i;
                int iw = iw_start + j;
                if (ih >= 0 && ih < (int)IH && iw >= 0 && iw < (int)IW) {
                    mid_buf1[i * alpha + j] =
                            getter(input[ic * IH * IW + ih * IW + iw]);
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
            size_t icb = ic / pack_size;
            size_t ic_pack = ic % pack_size;
            rep(i, alpha) rep(j, alpha) {
                input_transform_buf[(i * alpha + j) * ICB * nr_units_in_tile *
                                            pack_size +
                                    icb * nr_units_in_tile * pack_size +
                                    unit_idx * pack_size + ic_pack] =
                        mid_buf1[i * alpha + j];
            }
        }
    }

    static void output(const output_compute_type* output_transform_buf,
                       const output_compute_type* bias, dst_type* output,
                       output_compute_type* transform_mid_buf, BiasMode bmode,
                       NonlineMode nonline_mode, size_t oh_start,
                       size_t ow_start, size_t OH, size_t OW, size_t oc_start,
                       size_t oc_end, size_t unit_idx, size_t nr_units_in_tile,
                       size_t m, size_t r,
                       const std::vector<float>& interp_points, DType dtype,
                       float input_filter_scale, float input_filter_rescale,
                       float rescale) {
        size_t alpha = m + r - 1;
        size_t OC = oc_end - oc_start;

        OutputGetter<output_compute_type, dst_type> getter(dtype);
        winograd::WinogradCoeff<output_compute_type> winograd_coeff(
                m, r, interp_points);
        size_t OCB = OC / pack_size;
        for (size_t oc = oc_start; oc < oc_end; oc++) {
            output_compute_type* mid_buf1 = transform_mid_buf;
            output_compute_type* mid_buf2 = transform_mid_buf + alpha * alpha;

            size_t ocb = (oc - oc_start) / pack_size;
            size_t oc_pack = oc % pack_size;
            // gather
            rep(i, alpha) rep(j, alpha) {
                mid_buf1[i * alpha + j] = output_transform_buf
                        [(i * alpha + j) * OCB * nr_units_in_tile * pack_size +
                         ocb * nr_units_in_tile * pack_size +
                         unit_idx * pack_size + oc_pack];
            }
            /* A[alpha*m] M[alpha*alpha] */
            megdnn::naive::run_matrix_mul_tpl<output_compute_type,
                                              output_compute_type, true, false>(
                    winograd_coeff.A(rescale).data(), mid_buf1, mid_buf2, m,
                    alpha, alpha, m, alpha, alpha, dtype, dtype);
            megdnn::naive::run_matrix_mul_tpl<
                    output_compute_type, output_compute_type, false, false>(
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
                        val += bias[oc * OH * OW + oh * OW + ow] *
                               input_filter_rescale * input_filter_rescale;
                    }
                    val = val * input_filter_scale /
                          (input_filter_rescale * input_filter_rescale *
                           rescale * rescale);
                    if (nonline_mode == NonlineMode::RELU) {
                        val = val > 0 ? val : 0;
                    } else if (nonline_mode == NonlineMode::SIGMOID) {
                        val = 1.f / (expf(-val) + 1.f);
                    } else if (nonline_mode == NonlineMode::H_SWISH) {
                        val = val * std::min(std::max(val + 3, 0.f), 6.f) / 6.f;
                    } else {
                        megdnn_assert(nonline_mode == NonlineMode::IDENTITY);
                    }

                    output[oc * OH * OW + oh * OW + ow] = getter(val);
                }
            }
        }
    }

    static size_t pack_size;
};

template <typename ctype, typename dst_type, typename input_filter_compute_type,
          typename output_compute_type, param::MatrixMul::Format format>
size_t StrategyHelper<
        ctype, dst_type, input_filter_compute_type, output_compute_type, format,
        std::enable_if_t<format == param::MatrixMul::Format::MK4 ||
                         format == param::MatrixMul::Format::MK8>>::pack_size =
        MatrixMulForward::pack_size(format);

#define INST(_ctype, _dst_type, _input_filter_compute_type, \
             _output_compute_type)                          \
    template class StrategyHelper<                          \
            _ctype, _dst_type, _input_filter_compute_type,  \
            _output_compute_type, param::MatrixMul::Format::DEFAULT>;

INST(float, float, float, float)
MEGDNN_INC_FLOAT16(INST(dt_float16, dt_float16, dt_float16, dt_float16))
INST(int8_t, int8_t, int16_t, int)
INST(uint8_t, uint8_t, int16_t, int)
#undef INST

#define INST(_ctype, _dst_type, _input_filter_compute_type, \
             _output_compute_type)                          \
    template class StrategyHelper<                          \
            _ctype, _dst_type, _input_filter_compute_type,  \
            _output_compute_type, param::MatrixMul::Format::MK4>;
INST(float, float, float, float)
#undef INST

#define INST(_ctype, _dst_type, _input_filter_compute_type, \
             _output_compute_type)                          \
    template class StrategyHelper<                          \
            _ctype, _dst_type, _input_filter_compute_type,  \
            _output_compute_type, param::MatrixMul::Format::MK8>;
INST(int8_t, int8_t, int16_t, int)
MEGDNN_INC_FLOAT16(INST(dt_float16, dt_float16, dt_float16, dt_float16))
#undef INST

template <typename ctype, typename dst_type, typename input_filter_compute_type,
          typename output_compute_type, param::MatrixMul::Format format>
class StrategyHelperNchwxx<
        ctype, dst_type, input_filter_compute_type, output_compute_type, format,
        std::enable_if_t<format == param::MatrixMul::Format::MK8>> {
public:
    static void filter(const ctype* filter,
                       input_filter_compute_type* filter_transform_buf,
                       input_filter_compute_type* transform_mid_buf, size_t OC,
                       size_t IC, size_t oc_start, size_t oc_end, size_t m,
                       size_t r, const std::vector<float>& interp_points,
                       DType dtype, float rescale) {
        megdnn_assert(
                (oc_end - oc_start) % 8 == 0 && oc_start % 8 == 0 &&
                        oc_end % 8 == 0 && IC % 8 == 0 && OC % 8 == 0,
                "Winograd filter transform input param is not times of 8!");

        size_t alpha = m + r - 1;
        WinogradCoeff<input_filter_compute_type> winograd_coeff(m, r,
                                                                interp_points);

        input_filter_compute_type* mid_buf1 = transform_mid_buf;
        input_filter_compute_type* mid_buf2 = transform_mid_buf + alpha * alpha;

        Getter<ctype, input_filter_compute_type> getter(dtype);
        size_t OCB = OC / pack_size;
        size_t ICB = IC / pack_size;
        for (size_t oc = oc_start; oc < oc_end; oc++) {
            rep(ic, IC) {
                size_t ocb = oc / pack_size;
                size_t oc_pack = oc % pack_size;
                size_t icb = ic / pack_size;
                size_t ic_pack = ic % pack_size;

                const ctype* filter_ptr =
                        filter + (ocb * (IC / 8) + icb) * r * r * 8 * 8 +
                        ic_pack * 8 + oc_pack;
                rep(i, r) rep(j, r) {
                    mid_buf1[i * r + j] =
                            getter(filter_ptr[(i * r + j) * 8 * 8]);
                }

                /* tmp = Matmul(G, src) */
                megdnn::naive::run_matrix_mul_tpl<input_filter_compute_type,
                                                  input_filter_compute_type,
                                                  false, false>(
                        winograd_coeff.G(rescale).data(), mid_buf1, mid_buf2,
                        alpha, r, r, r, r, r, dtype, dtype);
                /* dst = Matmul(tmp, G^T) */
                megdnn::naive::run_matrix_mul_tpl<input_filter_compute_type,
                                                  input_filter_compute_type,
                                                  false, true>(
                        mid_buf2, winograd_coeff.G(rescale).data(), mid_buf1,
                        alpha, alpha, r, r, r, alpha, dtype, dtype);

                rep(i, alpha) rep(j, alpha) {
                    filter_transform_buf[(i * alpha + j) * OCB * ICB *
                                                 pack_size * pack_size +
                                         ocb * ICB * pack_size * pack_size +
                                         icb * pack_size * pack_size +
                                         ic_pack * pack_size + oc_pack] =
                            mid_buf1[i * alpha + j];
                }
            }
        }
    }

    static void input(const ctype* input,
                      input_filter_compute_type* input_transform_buf,
                      input_filter_compute_type* transform_mid_buf,
                      int ih_start, int iw_start, size_t IH, size_t IW,
                      size_t IC, size_t unit_idx, size_t nr_units_in_tile,
                      size_t m, size_t r,
                      const std::vector<float>& interp_points, DType dtype,
                      float rescale) {
        size_t alpha = m + r - 1;
        Getter<ctype, input_filter_compute_type> getter(dtype);
        WinogradCoeff<input_filter_compute_type> winograd_coeff(m, r,
                                                                interp_points);
        size_t ICB = IC / pack_size;
        rep(ic, IC) {
            size_t icb = ic / pack_size;
            size_t ic_pack = ic % pack_size;
            input_filter_compute_type* mid_buf1 = transform_mid_buf;
            input_filter_compute_type* mid_buf2 =
                    transform_mid_buf + alpha * alpha;

            memset(mid_buf1, 0,
                   alpha * alpha * sizeof(input_filter_compute_type));
            rep(i, alpha) rep(j, alpha) {
                int ih = ih_start + i;
                int iw = iw_start + j;
                if (ih >= 0 && ih < (int)IH && iw >= 0 && iw < (int)IW) {
                    mid_buf1[i * alpha + j] = getter(
                            input[(icb * IH * IW + ih * IW + iw) * pack_size +
                                  ic_pack]);
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
                input_transform_buf[(i * alpha + j) * ICB * nr_units_in_tile *
                                            pack_size +
                                    icb * nr_units_in_tile * pack_size +
                                    unit_idx * pack_size + ic_pack] =
                        mid_buf1[i * alpha + j];
            }
        }
    }

    static void output(const output_compute_type* output_transform_buf,
                       const output_compute_type* bias, dst_type* output,
                       output_compute_type* transform_mid_buf, BiasMode bmode,
                       NonlineMode nonline_mode, size_t oh_start,
                       size_t ow_start, size_t OH, size_t OW, size_t oc_start,
                       size_t oc_end, size_t unit_idx, size_t nr_units_in_tile,
                       size_t m, size_t r,
                       const std::vector<float>& interp_points, DType dtype,
                       float input_filter_scale, float input_filter_rescale,
                       float rescale) {
        size_t alpha = m + r - 1;
        size_t OC = oc_end - oc_start;

        OutputGetter<output_compute_type, dst_type> getter(dtype);
        winograd::WinogradCoeff<output_compute_type> winograd_coeff(
                m, r, interp_points);
        size_t OCB = OC / pack_size;
        for (size_t oc = oc_start; oc < oc_end; oc++) {
            output_compute_type* mid_buf1 = transform_mid_buf;
            output_compute_type* mid_buf2 = transform_mid_buf + alpha * alpha;

            size_t ocb = (oc - oc_start) / pack_size;
            size_t oc_pack = oc % pack_size;
            // gather
            rep(i, alpha) rep(j, alpha) {
                mid_buf1[i * alpha + j] = output_transform_buf
                        [(i * alpha + j) * OCB * nr_units_in_tile * pack_size +
                         ocb * nr_units_in_tile * pack_size +
                         unit_idx * pack_size + oc_pack];
            }
            /* A[alpha*m] M[alpha*alpha] */
            megdnn::naive::run_matrix_mul_tpl<output_compute_type,
                                              output_compute_type, true, false>(
                    winograd_coeff.A(rescale).data(), mid_buf1, mid_buf2, m,
                    alpha, alpha, m, alpha, alpha, dtype, dtype);
            megdnn::naive::run_matrix_mul_tpl<
                    output_compute_type, output_compute_type, false, false>(
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
                        val += bias[(oc / pack_size * OH * OW + oh * OW + ow) *
                                            pack_size +
                                    oc_pack] *
                               input_filter_rescale * input_filter_rescale;
                    }
                    val = val * input_filter_scale /
                          (input_filter_rescale * input_filter_rescale *
                           rescale * rescale);
                    if (nonline_mode == NonlineMode::RELU) {
                        val = val > 0 ? val : 0;
                    } else if (nonline_mode == NonlineMode::SIGMOID) {
                        val = 1.f / (expf(-val) + 1.f);
                    } else if (nonline_mode == NonlineMode::H_SWISH) {
                        val = val * std::min(std::max(val + 3, 0.f), 6.f) / 6.f;
                    } else {
                        megdnn_assert(nonline_mode == NonlineMode::IDENTITY);
                    }

                    output[(oc / pack_size * OH * OW + oh * OW + ow) *
                                   pack_size +
                           oc_pack] = getter(val);
                }
            }
        }
    }

    static size_t pack_size;
};

template <typename ctype, typename dst_type, typename input_filter_compute_type,
          typename output_compute_type, param::MatrixMul::Format format>
size_t StrategyHelperNchwxx<
        ctype, dst_type, input_filter_compute_type, output_compute_type, format,
        std::enable_if_t<format == param::MatrixMul::Format::MK8>>::pack_size =
        MatrixMulForward::pack_size(format);

#define INST(_ctype, _dst_type, _input_filter_compute_type, \
             _output_compute_type)                          \
    template class StrategyHelperNchwxx<                    \
            _ctype, _dst_type, _input_filter_compute_type,  \
            _output_compute_type, param::MatrixMul::Format::MK8>;
INST(float, float, float, float)
#undef INST



}  // namespace winograd
}  // namespace megdnn

// vim: syntax=cpp.doxygen
