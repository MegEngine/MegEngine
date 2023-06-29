#include "src/common/rounding_converter.cuh"
#include "src/common/utils.cuh"
#include "src/cuda/resize/common.cuh"
#include "src/cuda/resize/common.h"
#include "src/cuda/resize/resize_cv.cuh"

#include "src/common/resize.cuh"
#include "src/cuda/cv/kernel_common.cuh"
#include "src/cuda/kernel_common/diagnostic_prologue.cuh"

using namespace megdnn;
using namespace cuda;
using namespace megdnn::cuda::resize;
using megdnn::megcv::saturate;
using megdnn::resize::interpolate_cubic;

namespace {

template <typename ctype>
struct DirectSrcVisitor {
    const ctype* ptr;

    __device__ __forceinline__ const ctype* get(int batch, int im_size) {
        return ptr + batch * im_size;
    }

    void move_batch(size_t batch, size_t im_size) { ptr += batch * im_size; }
};

template <typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_general_linear(
        SrcVisitor src, ctype* __restrict dst, int C, int IH, int IW, int OH, int OW,
        int S_IN, int S_IC, int S_IH, int S_IW, float scale_h, float scale_w) {
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, S_IN);
    dst += blockIdx.z * C * OH * OW;

    if (ow < OW && oh < OH) {
        float alphah, alphaw;
        int ih0, iw0;
        get_origin_coord(scale_h, IH, oh, alphah, ih0);
        get_origin_coord(scale_w, IW, ow, alphaw, iw0);

        int ih1 = ih0 + 1;
        int iw1 = iw0 + 1;

        for (int c = 0; c < C; ++c) {
            dst[oh * OW + ow] = output_converter(
                    sptr[ih0 * S_IH + iw0 * S_IW] * (1.0f - alphaw) * (1.0f - alphah) +
                    sptr[ih0 * S_IH + iw1 * S_IW] * alphaw * (1.0f - alphah) +
                    sptr[ih1 * S_IH + iw0 * S_IW] * (1.0f - alphaw) * alphah +
                    sptr[ih1 * S_IH + iw1 * S_IW] * alphaw * alphah);

            sptr += S_IC;
            dst += OH * OW;
        }
    }
}

template <typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_general_nearest(
        SrcVisitor src, ctype* __restrict dst, int C, int IH, int IW, int OH, int OW,
        int S_IN, int S_IC, int S_IH, int S_IW, float scale_h, float scale_w) {
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, S_IN);
    dst += blockIdx.z * C * OH * OW;

    if (ow < OW && oh < OH) {
        int ih = get_nearest_src(scale_h, IH, oh);
        int iw = get_nearest_src(scale_w, IW, ow);

        for (int c = 0; c < C; ++c) {
            dst[oh * OW + ow] = output_converter(sptr[ih * S_IH + iw * S_IW]);

            sptr += S_IC;
            dst += OH * OW;
        }
    }
}

template <typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_general_cubic(
        SrcVisitor src, ctype* __restrict dst, int C, int IH, int IW, int OH, int OW,
        int S_IN, int S_IC, int S_IH, int S_IW, float scale_h, float scale_w) {
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, S_IN);
    dst += blockIdx.z * C * OH * OW;

    if (ow < OW && oh < OH) {
        float alphah, alphaw;
        int ih0, iw0;
        get_origin_coord(scale_h, IH, oh, alphah, ih0, true);
        get_origin_coord(scale_w, IW, ow, alphaw, iw0, true);
        ih0--;
        iw0--;
        float h_coeff[4], w_coeff[4];
        interpolate_cubic(alphah, h_coeff);
        interpolate_cubic(alphaw, w_coeff);
        for (int c = 0; c < C; ++c) {
            float ret = 0;
            constexpr int ksize = 4;
            for (int kh = 0; kh < ksize; kh++) {
                int ih = saturate(ih0 + kh, 0, IH - 1);
                for (int kw = 0; kw < ksize; kw++) {
                    int iw = saturate(iw0 + kw, 0, IW - 1);
                    ret += sptr[ih * S_IH + iw * S_IW] * h_coeff[kh] * w_coeff[kw];
                }
            }
            dst[oh * OW + ow] = output_converter(ret);

            sptr += S_IC;
            dst += OH * OW;
        }
    }
}
template <typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_general_nhwc(
        SrcVisitor src, ctype* __restrict dst, int C, int IH, int IW, int OH, int OW,
        float scale_h, float scale_w) {
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);
    dst += blockIdx.z * C * OH * OW;
    if (ow < OW && oh < OH) {
        float alphah, alphaw;
        int ih0, iw0;
        get_origin_coord(scale_h, IH, oh, alphah, ih0);
        get_origin_coord(scale_w, IW, ow, alphaw, iw0);

        int ih1 = ih0 + 1;
        int iw1 = iw0 + 1;

        for (int c = 0; c < C; ++c) {
            dst[(oh * OW + ow) * C + c] = output_converter(
                    sptr[(ih0 * IW + iw0) * C + c] * (1.0f - alphaw) * (1.0f - alphah) +
                    sptr[(ih0 * IW + iw1) * C + c] * alphaw * (1.0f - alphah) +
                    sptr[(ih1 * IW + iw0) * C + c] * (1.0f - alphaw) * alphah +
                    sptr[(ih1 * IW + iw1) * C + c] * alphaw * alphah);
        }
    }
}

template <typename ctype, typename SrcVisitor>
void dispatch_with_visitor(
        bool is_nhwc, InterpolationMode imode, SrcVisitor src, ctype* dst, int N, int C,
        int IH, int IW, int OH, int OW, int S_IN, int S_IC, int S_IH, int S_IW,
        cudaStream_t stream) {
    const int BY = 16, BX = 32;

    const int max_batch_size = 65535;
    while (N) {
        size_t curr_batch_size = N < max_batch_size ? N : max_batch_size;
        dim3 threads(BX, BY);
        dim3 blocks((OW + BX - 1) / BX, (OH + BY - 1) / BY, curr_batch_size);

        float scale_h = static_cast<float>(OH) / IH;
        float scale_w = static_cast<float>(OW) / IW;
        if (is_nhwc) {
            kern_general_nhwc<ctype, SrcVisitor, rounding::RoundingConverter<ctype>>
                    <<<blocks, threads, 0, stream>>>(
                            src, dst, C, IH, IW, OH, OW, scale_h, scale_w);
        } else {
            switch (imode) {
                case InterpolationMode::INTER_LINEAR:
                    kern_general_linear<
                            ctype, SrcVisitor, rounding::RoundingConverter<ctype>>
                            <<<blocks, threads, 0, stream>>>(
                                    src, dst, C, IH, IW, OH, OW, S_IN, S_IC, S_IH, S_IW,
                                    scale_h, scale_w);
                    break;
                case InterpolationMode::INTER_NEAREST:
                    kern_general_nearest<
                            ctype, SrcVisitor, rounding::RoundingConverter<ctype>>
                            <<<blocks, threads, 0, stream>>>(
                                    src, dst, C, IH, IW, OH, OW, S_IN, S_IC, S_IH, S_IW,
                                    scale_h, scale_w);
                    break;
                case InterpolationMode::INTER_CUBIC:
                    kern_general_cubic<
                            ctype, SrcVisitor, rounding::RoundingConverter<ctype>>
                            <<<blocks, threads, 0, stream>>>(
                                    src, dst, C, IH, IW, OH, OW, S_IN, S_IC, S_IH, S_IW,
                                    scale_h, scale_w);
                    break;
                default:
                    megdnn_throw("unsupported interpolation mode");
                    break;
            }
        }
        N -= curr_batch_size;
        src.move_batch(curr_batch_size, C * IH * IW);
        dst += curr_batch_size * C * OH * OW;
    }
}

template <typename ctype, typename SrcVisitor, typename OutputConverter>
__global__ void kern_general_nchw4(
        SrcVisitor src, ctype* __restrict dst, int C, int IH, int IW, int OH, int OW,
        float scale_h, float scale_w) {
    OutputConverter output_converter;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    const ctype* __restrict sptr = src.get(blockIdx.z, C * IH * IW);
    dst += blockIdx.z * C * OH * OW;

    if (ow < OW && oh < OH) {
        float alphah, alphaw;
        int ih0, iw0;
        get_origin_coord(scale_h, IH, oh, alphah, ih0);
        get_origin_coord(scale_w, IW, ow, alphaw, iw0);

        int ih1 = ih0 + 1;
        int iw1 = iw0 + 1;

        int o_coor = (oh * OW + ow) << 2;
        int i_coor00 = (ih0 * IW + iw0) << 2;
        int i_coor01 = (ih0 * IW + iw1) << 2;
        int i_coor10 = (ih1 * IW + iw0) << 2;
        int i_coor11 = (ih1 * IW + iw1) << 2;
        for (int c0 = 0, nr_chan = C >> 2; c0 < nr_chan; ++c0) {
#pragma unroll
            for (int c1 = 0; c1 < 4; ++c1) {
                dst[o_coor + c1] = output_converter(
                        sptr[i_coor00 + c1] * (1.0f - alphaw) * (1.0f - alphah) +
                        sptr[i_coor01 + c1] * alphaw * (1.0f - alphah) +
                        sptr[i_coor10 + c1] * (1.0f - alphaw) * alphah +
                        sptr[i_coor11 + c1] * alphaw * alphah);
            }
            dst += OH * OW * 4;
            sptr += IH * IW * 4;
        }
    }
}

template <typename ctype, typename SrcVisitor>
void dispatch_with_visitor_nchw4(
        SrcVisitor src, ctype* dst, int N, int C, int IH, int IW, int OH, int OW,
        cudaStream_t stream) {
    const int BY = 16, BX = 32;

    const int max_batch_size = 65535;
    while (N) {
        size_t curr_batch_size = N < max_batch_size ? N : max_batch_size;
        dim3 threads(BX, BY);
        dim3 blocks((OW + BX - 1) / BX, (OH + BY - 1) / BY, curr_batch_size);

        float scale_h = static_cast<float>(OH) / IH;
        float scale_w = static_cast<float>(OW) / IW;
        kern_general_nchw4<ctype, SrcVisitor, rounding::RoundingConverter<ctype>>
                <<<blocks, threads, 0, stream>>>(
                        src, dst, C, IH, IW, OH, OW, scale_h, scale_w);
        N -= curr_batch_size;
        src.move_batch(curr_batch_size, C * IH * IW);
        dst += curr_batch_size * C * OH * OW;
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace resize {

template <typename ctype>
void forward_proxy(
        bool is_nhwc, InterpolationMode imode, const ctype* src, ctype* dst, int N,
        int C, int IH, int IW, int OH, int OW, int S_IN, int S_IC, int S_IH, int S_IW,
        cudaStream_t stream) {
    DirectSrcVisitor<ctype> visitor;
    visitor.ptr = src;
    dispatch_with_visitor(
            is_nhwc, imode, visitor, dst, N, C, IH, IW, OH, OW, S_IN, S_IC, S_IH, S_IW,
            stream);
    after_kernel_launch();
}

template <typename ctype>
void forward_proxy_nchw4(
        const ctype* src, ctype* dst, int N, int C, int IH, int IW, int OH, int OW,
        cudaStream_t stream) {
    DirectSrcVisitor<ctype> visitor;
    visitor.ptr = src;
    dispatch_with_visitor_nchw4(visitor, dst, N, C, IH, IW, OH, OW, stream);
    after_kernel_launch();
}

#define INST(ctype)                                                                 \
    template void forward_proxy(                                                    \
            bool, InterpolationMode, const ctype*, ctype*, int, int, int, int, int, \
            int, int, int, int, int, cudaStream_t);
INST(float)
INST(uint8_t)
INST(int8_t)
DNN_INC_FLOAT16(INST(dt_float16))
#undef INST

#define INST(ctype)                    \
    template void forward_proxy_nchw4( \
            const ctype*, ctype*, int, int, int, int, int, int, cudaStream_t)

INST(int8_t);
#undef INST
}  // namespace resize

namespace resize3d {

__device__ __forceinline__ static float pixel_get_src_index(
        float scale, int64_t dst_index, bool align_corners) {
    if (align_corners) {
        return scale * dst_index;
    } else {
        float src_idx = scale * (dst_index + 0.5f) - 0.5f;
        return src_idx < 0.f ? 0.f : src_idx;
    }
}

__device__ __forceinline__ static size_t index_getter(
        int n, int c, int d, int h, int w, const int N, const int C, const int D,
        const int H, const int W) {
    return n * C * D * H * W + c * D * H * W + d * H * W + h * W + w;
}

template <typename ctype>
__global__ void trilinear_forward(
        const int num_kernels, const float rdepth, const float rheight,
        const float rwidth, const bool align_corners, const ctype* iptr, ctype* optr,
        const int N, const int C, const int ID, const int IH, const int IW,
        const int OD, const int OH, const int OW) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < num_kernels) {
        const int w2 = (index % (OH * OW)) % OW;
        const int h2 = (index % (OH * OW)) / OW;
        const int t2 = index / (OH * OW);

        if (ID == OD && IH == OH && IW == OW) {
            const int t1 = t2;
            const int h1 = h2;
            const int w1 = w2;

            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; ++c) {
                    const ctype val =
                            iptr[index_getter(n, c, t1, h1, w1, N, C, ID, IH, IW)];
                    optr[index_getter(n, c, t2, h2, w2, N, C, OD, OH, OW)] = val;
                }
            }
            return;
        }

        const float t1r = pixel_get_src_index(rdepth, t2, align_corners);
        const int t1 = t1r;
        const int t1p = (t1 < ID - 1) ? 1 : 0;
        const float t1lambda = t1r - t1;
        const float t0lambda = static_cast<float>(1) - t1lambda;

        const float h1r = pixel_get_src_index(rheight, h2, align_corners);
        const int h1 = h1r;
        const int h1p = (h1 < IH - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = static_cast<float>(1) - h1lambda;

        const float w1r = pixel_get_src_index(rwidth, w2, align_corners);
        const int w1 = w1r;
        const int w1p = (w1 < IW - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = static_cast<float>(1) - w1lambda;

        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; ++c) {
                const float val =
                        t0lambda *
                                (h0lambda * (w0lambda * iptr[index_getter(
                                                                n, c, t1, h1, w1, N, C,
                                                                ID, IH, IW)] +
                                             w1lambda * iptr[index_getter(
                                                                n, c, t1, h1, w1 + w1p,
                                                                N, C, ID, IH, IW)]) +
                                 h1lambda *
                                         (w0lambda * iptr[index_getter(
                                                             n, c, t1, h1 + h1p, w1, N,
                                                             C, ID, IH, IW)] +
                                          w1lambda *
                                                  iptr[index_getter(
                                                          n, c, t1, h1 + h1p, w1 + w1p,
                                                          N, C, ID, IH, IW)])) +
                        t1lambda *
                                (h0lambda *
                                         (w0lambda * iptr[index_getter(
                                                             n, c, t1 + t1p, h1, w1, N,
                                                             C, ID, IH, IW)] +
                                          w1lambda *
                                                  iptr[index_getter(
                                                          n, c, t1 + t1p, h1, w1 + w1p,
                                                          N, C, ID, IH, IW)]) +
                                 h1lambda *
                                         (w0lambda * iptr[index_getter(
                                                             n, c, t1 + t1p, h1 + h1p,
                                                             w1, N, C, ID, IH, IW)] +
                                          w1lambda * iptr[index_getter(
                                                             n, c, t1 + t1p, h1 + h1p,
                                                             w1 + w1p, N, C, ID, IH,
                                                             IW)]));
                optr[index_getter(n, c, t2, h2, w2, N, C, OD, OH, OW)] =
                        static_cast<ctype>(val);
            }
        }
    }
}

__host__ __forceinline__ static float get_scale(
        int input_size, int output_size, bool align_corners) {
    if (align_corners) {
        if (output_size > 1) {
            return static_cast<float>(input_size - 1) / (output_size - 1);
        } else {
            return 0.f;
        }
    } else {
        return static_cast<float>(input_size) / output_size;
    }
}

template <typename ctype>
void resize3d_forward(
        const bool align_corners, const ctype* iptr, ctype* optr, const int N,
        const int C, const int ID, const int IH, const int IW, const int OD,
        const int OH, const int OW, cudaStream_t stream) {
    const size_t num_kernels = OD * OH * OW;
    const size_t num_threads = 512;

    float rdepth = get_scale(ID, OD, align_corners);
    float rheight = get_scale(IH, OH, align_corners);
    float rwidth = get_scale(IW, OW, align_corners);

    trilinear_forward<ctype>
            <<<(num_kernels + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
                    num_kernels, rdepth, rheight, rwidth, align_corners, iptr, optr, N,
                    C, ID, IH, IW, OD, OH, OW);
}

#define INST(ctype)                                                            \
    template void resize3d_forward(                                            \
            const bool, const ctype*, ctype*, const int, const int, const int, \
            const int, const int, const int, const int, const int, cudaStream_t);

INST(float)
DNN_INC_FLOAT16(INST(dt_float16))

#undef INST

}  // namespace resize3d

}  // namespace cuda
}  // namespace megdnn

#include "src/cuda/kernel_common/diagnostic_epilogue.cuh"
// vim: syntax=cpp.doxygen
