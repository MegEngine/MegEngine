/**
 * \file dnn/src/naive/remap/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/naive/remap/opr_impl.h"
#include "src/common/cv/helper.h"
#include "src/common/rounding_converter.cuh"
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;
using namespace rounding;

namespace {
template <param::Remap::Format format>
inline int get_offset(int, int, int, int, int, int);

template <>
inline int get_offset<param::Remap::Format::NCHW>(int height, int width,
                                                  int channel, int h, int w,
                                                  int) {
    return channel * h * w + height * w + width;
}

template <>
inline int get_offset<param::Remap::Format::NHWC>(int height, int width,
                                                  int channel, int, int w,
                                                  int c) {
    return height * w * c + width * c + channel;
}

template <typename ctype, param::Remap::Format format,
          param::Remap::BorderMode bordertype>
struct GetSrcData {
    static inline ctype get(const ctype* src, int height, int width,
                            int channel, int h, int w, int c, float) {
        height = megcv::border_interpolate<bordertype>(height, h);
        width = megcv::border_interpolate<bordertype>(width, w);
        return src[get_offset<format>(height, width, channel, h, w, c)];
    }
    static inline int get_index(int height, int width, int channel, int h,
                                int w, int c) {
        height = megcv::border_interpolate<bordertype>(height, h);
        width = megcv::border_interpolate<bordertype>(width, w);
        return get_offset<format>(height, width, channel, h, w, c);
    }
};

template <typename ctype, param::Remap::Format format>
struct GetSrcData<ctype, format, param::Remap::BorderMode::CONSTANT> {
    static inline ctype get(const ctype* src, int height, int width,
                            int channel, int h, int w, int c, float scalar) {
        RoundingConverter<ctype> round;
        return (height >= 0 && height < h && width >= 0 && width < w)
                       ? src[get_offset<format>(height, width, channel, h, w,
                                                c)]
                       : round(scalar);
    }
    static inline int get_index(int height, int width, int channel, int h,
                                int w, int c) {
        return (height >= 0 && height < h && width >= 0 && width < w)
                       ? get_offset<format>(height, width, channel, h, w, c)
                       : -1;
    }
};

template <typename ctype, param::Remap::Format format,
          param::Remap::BorderMode bordertype>
void remap_LINEAR(const ctype* src, const float* map_xy, ctype* dst, int N,
                  int C, int IH, int IW, int OH, int OW, float scalar) {
    RoundingConverter<ctype> round_converter;
    for (int n = 0; n < N;
         ++n, src += C * IH * IW, dst += C * OH * OW, map_xy += OH * OW * 2) {
        for (int h = 0; h < OH; ++h) {
            for (int w = 0; w < OW; ++w) {
                float index_col = map_xy[h * OW * 2 + w * 2 + 0];
                float index_row = map_xy[h * OW * 2 + w * 2 + 1];
                int col = static_cast<int>(floor(index_col));
                int row = static_cast<int>(floor(index_row));
                float v = index_col - col;  // alphaw
                float u = index_row - row;  // alphah
                const float one = 1.f;
                for (int c = 0; c < C; ++c) {
                    ctype a00 = GetSrcData<ctype, format, bordertype>::get(
                            src, row + 0, col + 0, c, IH, IW, C, scalar);
                    ctype a01 = GetSrcData<ctype, format, bordertype>::get(
                            src, row + 0, col + 1, c, IH, IW, C, scalar);
                    ctype a10 = GetSrcData<ctype, format, bordertype>::get(
                            src, row + 1, col + 0, c, IH, IW, C, scalar);
                    ctype a11 = GetSrcData<ctype, format, bordertype>::get(
                            src, row + 1, col + 1, c, IH, IW, C, scalar);

                    dst[get_offset<format>(h, w, c, OH, OW, C)] =
                            round_converter(a00 * (one - v) * (one - u) +
                                            a01 * (one - u) * v +
                                            a10 * (one - v) * u + a11 * u * v);
                }
            }
        }
    }
}

template <typename ctype, param::Remap::Format format,
          param::Remap::BorderMode bordertype>
void remap_LINEAR_backwarddata(ctype* grad, const float* map_xy,
                               const ctype* diff, int N, int C, int IH, int IW,
                               int OH, int OW) {
    RoundingConverter<ctype> round_converter;
    std::memset(grad, 0, sizeof(ctype) * N * C * IH * IW);
    for (int n = 0; n < N;
         ++n, grad += C * IH * IW, diff += C * OH * OW, map_xy += OH * OW * 2) {
        for (int h = 0; h < OH; ++h) {
            for (int w = 0; w < OW; ++w) {
                float index_col = map_xy[h * OW * 2 + w * 2 + 0];
                float index_row = map_xy[h * OW * 2 + w * 2 + 1];
                int col = static_cast<int>(floor(index_col));
                int row = static_cast<int>(floor(index_row));
                float v = index_col - col;  // alphaw
                float u = index_row - row;  // alphah
                const float one = 1.f;
                for (int c = 0; c < C; ++c) {
                    ctype hidden = diff[get_offset<format>(h, w, c, OH, OW, C)];

                    int a00 = GetSrcData<ctype, format, bordertype>::get_index(
                            row + 0, col + 0, c, IH, IW, C);
                    if (a00 != -1) {
                        grad[a00] +=
                                round_converter((one - v) * (one - u) * hidden);
                    }

                    int a01 = GetSrcData<ctype, format, bordertype>::get_index(
                            row + 0, col + 1, c, IH, IW, C);
                    if (a01 != -1) {
                        grad[a01] += round_converter((one - u) * v * hidden);
                    }

                    int a10 = GetSrcData<ctype, format, bordertype>::get_index(
                            row + 1, col + 0, c, IH, IW, C);
                    if (a10 != -1) {
                        grad[a10] += round_converter(u * (one - v) * hidden);
                    }

                    int a11 = GetSrcData<ctype, format, bordertype>::get_index(
                            row + 1, col + 1, c, IH, IW, C);
                    if (a11 != -1) {
                        grad[a11] += round_converter(v * u * hidden);
                    }
                }
            }
        }
    }
}

template <typename ctype, param::Remap::Format format,
          param::Remap::BorderMode bordertype>
void remap_LINEAR_backwardmat(const ctype* src, const float* map_xy,
                              const ctype* diff, float* grad, int N, int C,
                              int IH, int IW, int OH, int OW, float scalar) {
    RoundingConverter<ctype> round_converter;
    std::memset(grad, 0, sizeof(float) * N * 2 * OH * OW);
    for (int n = 0; n < N; ++n, src += C * IH * IW, diff += C * OH * OW,
             map_xy += OH * OW * 2, grad += OH * OW * 2) {
        for (int h = 0; h < OH; ++h) {
            for (int w = 0; w < OW; ++w) {
                float index_col = map_xy[h * OW * 2 + w * 2 + 0];
                float index_row = map_xy[h * OW * 2 + w * 2 + 1];
                int col = static_cast<int>(floor(index_col));
                int row = static_cast<int>(floor(index_row));
                float v = index_col - col;  // alphaw
                float u = index_row - row;  // alphah
                const float one = 1.f;
                for (int c = 0; c < C; ++c) {
                    float hidden = static_cast<float>(
                            diff[get_offset<format>(h, w, c, OH, OW, C)]);
                    float du = 0.f, dv = 0.f;

                    int a00 = GetSrcData<ctype, format, bordertype>::get_index(
                            row + 0, col + 0, c, IH, IW, C);
                    int a01 = GetSrcData<ctype, format, bordertype>::get_index(
                            row + 0, col + 1, c, IH, IW, C);
                    int a10 = GetSrcData<ctype, format, bordertype>::get_index(
                            row + 1, col + 0, c, IH, IW, C);
                    int a11 = GetSrcData<ctype, format, bordertype>::get_index(
                            row + 1, col + 1, c, IH, IW, C);

                    dv -= ((a00 != -1) ? src[a00] : scalar) * (one - u);
                    dv += ((a01 != -1) ? src[a01] : scalar) * (one - u);
                    dv -= ((a10 != -1) ? src[a10] : scalar) * u;
                    dv += ((a11 != -1) ? src[a11] : scalar) * u;
                    
                    du -= ((a00 != -1) ? src[a00] : scalar) * (one - v);
                    du -= ((a01 != -1) ? src[a01] : scalar) * v;
                    du += ((a10 != -1) ? src[a10] : scalar) * (one - v);
                    du += ((a11 != -1) ? src[a11] : scalar) * v;

                    grad[h * OW * 2 + w * 2 + 0] +=
                            round_converter(hidden * dv);
                    grad[h * OW * 2 + w * 2 + 1] +=
                            round_converter(hidden * du);
                }
            }
        }
    }
}

}  // namespace

void RemapImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in map_xy,
                     _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, map_xy.layout, dst.layout, workspace.size);
    int N, C, IH, IW, OH, OW;
    if (param().format == param::Remap::Format::NCHW) {
        N = src.layout.shape[0];
        C = src.layout.shape[1];
        IH = src.layout.shape[2];
        IW = src.layout.shape[3];
    } else if (param().format == param::Remap::Format::NHWC) {
        N = src.layout.shape[0];
        C = src.layout.shape[3];
        IH = src.layout.shape[1];
        IW = src.layout.shape[2];
    } else {
        megdnn_throw("unsupported format");
    }
    OH = map_xy.layout.shape[1];
    OW = map_xy.layout.shape[2];
    switch (src.layout.dtype.enumv()) {
#define cb(dt, fmt, border, interpolation)                                 \
    if (param().format == param::Remap::Format::fmt &&                     \
        param().border_type == param::Remap::BorderMode::border &&         \
        param().imode == param::Remap::InterpolationMode::interpolation) { \
        using ctype = DTypeTrait<dt>::ctype;                               \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                      \
                (remap_##interpolation<ctype, param::Remap::Format::fmt,   \
                                       param::Remap::BorderMode::border>(  \
                        src.compatible_ptr<ctype>(),                       \
                        map_xy.compatible_ptr<dt_float32>(),               \
                        dst.compatible_ptr<ctype>(), N, C, IH, IW, OH, OW, \
                        param().scalar)));                                 \
        break;                                                             \
    }

#define support_dtype(dt)                                                   \
    case DTypeTrait<dt>::enumv: {                                           \
        cb(dt, NCHW, CONSTANT, LINEAR);                                     \
        cb(dt, NCHW, REPLICATE, LINEAR);                                    \
        cb(dt, NCHW, REFLECT, LINEAR);                                      \
        cb(dt, NCHW, REFLECT_101, LINEAR);                                  \
        cb(dt, NCHW, WRAP, LINEAR);                                         \
        cb(dt, NHWC, CONSTANT, LINEAR);                                     \
        cb(dt, NHWC, REPLICATE, LINEAR);                                    \
        cb(dt, NHWC, REFLECT, LINEAR);                                      \
        cb(dt, NHWC, REFLECT_101, LINEAR);                                  \
        cb(dt, NHWC, WRAP, LINEAR);                                         \
        megdnn_throw(                                                       \
                "format, border type or imode is incorrect in remap navie " \
                "with dtype = " #dt);                                       \
    }

        support_dtype(dtype::Float32);
        MEGDNN_INC_FLOAT16(support_dtype(dtype::Float16));
        MEGDNN_INC_FLOAT16(support_dtype(dtype::BFloat16));
        support_dtype(dtype::Int8);
        support_dtype(dtype::Uint8);
#undef cb
#undef support_dtype

        default:
            megdnn_throw("unsupported dtype in remap naive\n");
    }
}

void RemapBackwardDataImpl::exec(_megdnn_tensor_in map_xy,
                                 _megdnn_tensor_in diff,
                                 _megdnn_tensor_out grad,
                                 _megdnn_workspace workspace) {
    check_exec(map_xy.layout, diff.layout, grad.layout, workspace.size);
    megdnn_assert(param().format == param::Remap::Format::NCHW,
                  "only support NCHW format for remap backward");
    int N, C, IH, IW, OH, OW;
    N = grad.layout.shape[0];
    C = grad.layout.shape[1];
    IH = grad.layout.shape[2];
    IW = grad.layout.shape[3];
    OH = map_xy.layout.shape[1];
    OW = map_xy.layout.shape[2];
    switch (diff.layout.dtype.enumv()) {
#define cb(dt, fmt, border, interpolation)                                  \
    if (param().format == param::Remap::Format::fmt &&                      \
        param().border_type == param::Remap::BorderMode::border &&          \
        param().imode == param::Remap::InterpolationMode::interpolation) {  \
        using ctype = DTypeTrait<dt>::ctype;                                \
        MEGDNN_DISPATCH_CPU_KERN_OPR((remap_##interpolation##_backwarddata< \
                                      ctype, param::Remap::Format::fmt,     \
                                      param::Remap::BorderMode::border>(    \
                grad.compatible_ptr<ctype>(),                               \
                map_xy.compatible_ptr<dt_float32>(),                        \
                diff.compatible_ptr<ctype>(), N, C, IH, IW, OH, OW)));      \
        break;                                                              \
    }

#define support_dtype(dt)                                                   \
    case DTypeTrait<dt>::enumv: {                                           \
        cb(dt, NCHW, CONSTANT, LINEAR);                                     \
        cb(dt, NCHW, REPLICATE, LINEAR);                                    \
        cb(dt, NCHW, REFLECT, LINEAR);                                      \
        cb(dt, NCHW, REFLECT_101, LINEAR);                                  \
        cb(dt, NCHW, WRAP, LINEAR);                                         \
        megdnn_throw(                                                       \
                "format, border type or imode is incorrect in remap navie " \
                "with dtype = " #dt);                                       \
    }

        support_dtype(dtype::Float32);
        MEGDNN_INC_FLOAT16(support_dtype(dtype::BFloat16));
#undef cb
#undef support_dtype

        default:
            megdnn_throw("unsupported dtype in remap backward naive\n");
    }
}

void RemapBackwardMatImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in map_xy,
                                _megdnn_tensor_in diff, _megdnn_tensor_out grad,
                                _megdnn_workspace workspace) {
    check_exec(src.layout, map_xy.layout, diff.layout, grad.layout,
               workspace.size);
    megdnn_assert(param().format == param::Remap::Format::NCHW,
                  "only support NCHW format for remap backward");
    int N, C, IH, IW, OH, OW;
    N = src.layout.shape[0];
    C = src.layout.shape[1];
    IH = src.layout.shape[2];
    IW = src.layout.shape[3];
    OH = map_xy.layout.shape[1];
    OW = map_xy.layout.shape[2];
    switch (src.layout.dtype.enumv()) {
#define cb(dt, fmt, border, interpolation)                                 \
    if (param().format == param::Remap::Format::fmt &&                     \
        param().border_type == param::Remap::BorderMode::border &&         \
        param().imode == param::Remap::InterpolationMode::interpolation) { \
        using ctype = DTypeTrait<dt>::ctype;                               \
        MEGDNN_DISPATCH_CPU_KERN_OPR((remap_##interpolation##_backwardmat< \
                                      ctype, param::Remap::Format::fmt,    \
                                      param::Remap::BorderMode::border>(   \
                src.compatible_ptr<ctype>(),                               \
                map_xy.compatible_ptr<dt_float32>(),                       \
                diff.compatible_ptr<ctype>(),                              \
                grad.compatible_ptr<dt_float32>(), N, C, IH, IW, OH, OW,   \
                param().scalar)));                                         \
        break;                                                             \
    }

#define support_dtype(dt)                                                   \
    case DTypeTrait<dt>::enumv: {                                           \
        cb(dt, NCHW, CONSTANT, LINEAR);                                     \
        cb(dt, NCHW, REPLICATE, LINEAR);                                    \
        cb(dt, NCHW, REFLECT, LINEAR);                                      \
        cb(dt, NCHW, REFLECT_101, LINEAR);                                  \
        cb(dt, NCHW, WRAP, LINEAR);                                         \
        megdnn_throw(                                                       \
                "format, border type or imode is incorrect in remap navie " \
                "with dtype = " #dt);                                       \
    }

        support_dtype(dtype::Float32);
        MEGDNN_INC_FLOAT16(support_dtype(dtype::BFloat16));
#undef cb
#undef support_dtype

        default:
            megdnn_throw("unsupported dtype in remap backward naive\n");
    }
}

// vim: syntax=cpp.doxygen
