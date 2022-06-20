#include "src/common/rounding_converter.cuh"
#include "src/cuda/resize/common.cuh"
#include "src/cuda/resize/common.h"

#include "src/cuda/cv/kernel_common.cuh"
#include "src/cuda/utils.cuh"

using megdnn::megcv::saturate;
using megdnn::resize::interpolate_cubic;

namespace megdnn {
namespace cuda {
namespace resize {

template <typename ctype, typename OutputConverter>
__global__ void resize_bwd_nhwc_kernel(
        const ctype* hidden, ctype* dst, int N, int C, int IH, int IW, int OH, int OW,
        float scale_h, float scale_w) {
    OutputConverter output_converter;
    int n = blockIdx.z;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    hidden += n * C * OH * OW;
    dst += n * C * IH * IW;
    if (ow < OW && oh < OH) {
        float alphah, alphaw;
        int ih0, iw0;
        get_origin_coord(scale_h, IH, oh, alphah, ih0);
        get_origin_coord(scale_w, IW, ow, alphaw, iw0);

        int ih1 = ih0 + 1;
        int iw1 = iw0 + 1;

        float nalphaw = 1.0f - alphaw;
        float nalphah = 1.0f - alphah;
        for (int c = 0; c < C; ++c) {
            atomic_add(
                    dst + (ih0 * IW + iw0) * C + c,
                    output_converter(
                            hidden[(oh * OW + ow) * C + c] * nalphaw * nalphah));
            atomic_add(
                    dst + (ih0 * IW + iw1) * C + c,
                    output_converter(
                            hidden[(oh * OW + ow) * C + c] * alphaw * nalphah));
            atomic_add(
                    dst + (ih1 * IW + iw0) * C + c,
                    output_converter(
                            hidden[(oh * OW + ow) * C + c] * nalphaw * alphah));
            atomic_add(
                    dst + (ih1 * IW + iw1) * C + c,
                    output_converter(hidden[(oh * OW + ow) * C + c] * alphaw * alphah));
        }
    }
}

template <typename ctype, typename OutputConverter>
__global__ void resize_bwd_linear_kernel(
        const ctype* hidden, ctype* dst, int N, int C, int IH, int IW, int OH, int OW,
        float scale_h, float scale_w) {
    OutputConverter output_converter;
    int n = blockIdx.z;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    hidden += n * C * OH * OW;
    dst += n * C * IH * IW;
    if (ow < OW && oh < OH) {
        float alphah, alphaw;
        int ih0, iw0;
        get_origin_coord(scale_h, IH, oh, alphah, ih0);
        get_origin_coord(scale_w, IW, ow, alphaw, iw0);

        int ih1 = ih0 + 1;
        int iw1 = iw0 + 1;

        float nalphaw = 1.0f - alphaw;
        float nalphah = 1.0f - alphah;
        for (int c = 0; c < C; ++c) {
            atomic_add(
                    dst + ih0 * IW + iw0,
                    output_converter(hidden[oh * OW + ow] * nalphaw * nalphah));
            atomic_add(
                    dst + ih0 * IW + iw1,
                    output_converter(hidden[oh * OW + ow] * alphaw * nalphah));
            atomic_add(
                    dst + ih1 * IW + iw0,
                    output_converter(hidden[oh * OW + ow] * nalphaw * alphah));
            atomic_add(
                    dst + ih1 * IW + iw1,
                    output_converter(hidden[oh * OW + ow] * alphaw * alphah));
            hidden += OH * OW;
            dst += IH * IW;
        }
    }
}

template <typename ctype, typename OutputConverter>
__global__ void resize_bwd_nearest_kernel(
        const ctype* hidden, ctype* dst, int N, int C, int IH, int IW, int OH, int OW,
        float scale_h, float scale_w) {
    OutputConverter output_converter;
    int n = blockIdx.z;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    hidden += n * C * OH * OW;
    dst += n * C * IH * IW;
    if (ow < OW && oh < OH) {
        int ih = get_nearest_src(scale_h, IH, oh);
        int iw = get_nearest_src(scale_w, IW, ow);

        for (int c = 0; c < C; ++c) {
            atomic_add(dst + ih * IW + iw, output_converter(hidden[oh * OW + ow]));
            hidden += OH * OW;
            dst += IH * IW;
        }
    }
}

template <typename ctype, typename OutputConverter>
__global__ void resize_bwd_cubic_kernel(
        const ctype* hidden, ctype* dst, int N, int C, int IH, int IW, int OH, int OW,
        float scale_h, float scale_w) {
    OutputConverter output_converter;
    int n = blockIdx.z;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    hidden += n * C * OH * OW;
    dst += n * C * IH * IW;
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
            constexpr int ksize = 4;
            for (int kh = 0; kh < ksize; kh++) {
                int ih = saturate(ih0 + kh, 0, IH - 1);
                for (int kw = 0; kw < ksize; kw++) {
                    int iw = saturate(iw0 + kw, 0, IW - 1);
                    atomic_add(
                            dst + ih * IW + iw,
                            output_converter(
                                    hidden[oh * OW + ow] * h_coeff[kh] * w_coeff[kw]));
                }
            }

            hidden += OH * OW;
            dst += IH * IW;
        }
    }
}

template <typename ctype>
void backward_data_proxy(
        bool is_nhwc, InterpolationMode imode, const ctype* diff, ctype* grad, int N,
        int C, int IH, int IW, int OH, int OW, cudaStream_t stream) {
    const int BY = 16, BX = 32;
    {
        dim3 threads(BX, BY);
        dim3 blocks((OW + BX - 1) / BX, (OH + BY - 1) / BY, N);
        cuda_check(cudaMemsetAsync(grad, 0, sizeof(ctype) * N * C * IH * IW, stream));
        float scale_h = static_cast<float>(OH) / IH;
        float scale_w = static_cast<float>(OW) / IW;
        if (is_nhwc) {
            resize_bwd_nhwc_kernel<ctype, rounding::RoundingConverter<ctype>>
                    <<<blocks, threads, 0, stream>>>(
                            diff, grad, N, C, IH, IW, OH, OW, scale_h, scale_w);
        } else {
            switch (imode) {
                case InterpolationMode::INTER_LINEAR: {
                    resize_bwd_linear_kernel<ctype, rounding::RoundingConverter<ctype>>
                            <<<blocks, threads, 0, stream>>>(
                                    diff, grad, N, C, IH, IW, OH, OW, scale_h, scale_w);
                    break;
                }
                case InterpolationMode::INTER_NEAREST: {
                    resize_bwd_nearest_kernel<ctype, rounding::RoundingConverter<ctype>>
                            <<<blocks, threads, 0, stream>>>(
                                    diff, grad, N, C, IH, IW, OH, OW, scale_h, scale_w);
                    break;
                }
                case InterpolationMode::INTER_CUBIC: {
                    resize_bwd_cubic_kernel<ctype, rounding::RoundingConverter<ctype>>
                            <<<blocks, threads, 0, stream>>>(
                                    diff, grad, N, C, IH, IW, OH, OW, scale_h, scale_w);
                    break;
                }
                default: {
                    megdnn_throw("unsupported interpolation mode");
                    break;
                }
            }
        }
    }
    after_kernel_launch();
}

#define INST(ctype)                                                                 \
    template void backward_data_proxy(                                              \
            bool, InterpolationMode, const ctype*, ctype*, int, int, int, int, int, \
            int, cudaStream_t);
INST(dt_float32);
DNN_INC_FLOAT16(INST(dt_float16));
#undef INST

}  // namespace resize
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
