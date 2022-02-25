#include <cuda.h>
#include <cuda_runtime.h>
#include "src/common/rounding_converter.cuh"
#include "src/cuda/cv/kernel_common.cuh"
#include "src/cuda/remap/common.h"
#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace cuda;
using namespace remap;
using namespace rounding;

namespace {

template <const uint32_t format>
__device__ inline int get_offset(
        int height, int width, int channel, int h, int w, int c);

template <>
__device__ inline int get_offset<param_enumv::Remap::Format::NCHW>(
        int height, int width, int channel, int h, int w, int c) {
    return channel * h * w + height * w + width;
}

template <>
__device__ inline int get_offset<param_enumv::Remap::Format::NHWC>(
        int height, int width, int channel, int h, int w, int c) {
    return height * w * c + width * c + channel;
}

template <typename ctype, const uint32_t format, ::BorderMode bmode>
struct GetSrcData {
    __device__ static inline ctype get(
            const ctype* src, int height, int width, int channel, int h, int w, int c,
            float) {
        height = megcv::border_interpolate<bmode>(height, h);
        width = megcv::border_interpolate<bmode>(width, w);
        return src[get_offset<format>(height, width, channel, h, w, c)];
    }
};

template <typename ctype, const uint32_t format>
struct GetSrcData<ctype, format, ::BorderMode::BORDER_CONSTANT> {
    __device__ static inline ctype get(
            const ctype* src, int height, int width, int channel, int h, int w, int c,
            float scalar) {
        RoundingConverter<ctype> round_converter;
        return (height >= 0 && height < h && width >= 0 && width < w)
                     ? src[get_offset<format>(height, width, channel, h, w, c)]
                     : round_converter(scalar);
    }
};

__device__ inline float round_half_to_even(float f) {
    const float round_away_from_zero = round(f);
    const float diff = round_away_from_zero - f;

    if ((diff != 0.5f) && (diff != -0.5f)) {
        return round_away_from_zero;
    }

    if (fmod(round_away_from_zero, 2.0f) == 0.0f) {
        return round_away_from_zero;
    }

    return f - diff;
}

template <typename ctype, const uint32_t format, ::BorderMode bmode>
__global__ void kern_general_nearest(
        const ctype* __restrict sptr, const float* map_xy, ctype* __restrict dst, int C,
        int IH, int IW, int OH, int OW, float scalar) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    sptr += blockIdx.z * C * IH * IW;
    dst += blockIdx.z * C * OH * OW;
    map_xy += blockIdx.z * 2 * OH * OW;

    if (ow < OW && oh < OH) {
        float index_col = map_xy[oh * OW * 2 + ow * 2 + 0];
        float index_row = map_xy[oh * OW * 2 + ow * 2 + 1];
        int col = static_cast<int>(round_half_to_even(index_col));
        int row = static_cast<int>(round_half_to_even(index_row));
        for (int c = 0; c < C; ++c) {
            dst[get_offset<format>(oh, ow, c, OH, OW, C)] =
                    GetSrcData<ctype, format, bmode>::get(
                            sptr, row, col, c, IH, IW, C, scalar);
        }
    }
}

template <typename ctype, const uint32_t format, ::BorderMode bmode>
__global__ void kern_general_linear(
        const ctype* __restrict sptr, const float* map_xy, ctype* __restrict dst, int C,
        int IH, int IW, int OH, int OW, float scalar) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    sptr += blockIdx.z * C * IH * IW;
    dst += blockIdx.z * C * OH * OW;
    map_xy += blockIdx.z * 2 * OH * OW;
    RoundingConverter<ctype> round_converter;

    if (ow < OW && oh < OH) {
        float index_col = map_xy[oh * OW * 2 + ow * 2 + 0];
        float index_row = map_xy[oh * OW * 2 + ow * 2 + 1];
        int col = static_cast<int>(floor(index_col));
        int row = static_cast<int>(floor(index_row));
        float v = index_col - col;
        float u = index_row - row;
        for (int c = 0; c < C; ++c) {
            ctype a00 = GetSrcData<ctype, format, bmode>::get(
                    sptr, row + 0, col + 0, c, IH, IW, C, scalar);
            ctype a01 = GetSrcData<ctype, format, bmode>::get(
                    sptr, row + 0, col + 1, c, IH, IW, C, scalar);
            ctype a10 = GetSrcData<ctype, format, bmode>::get(
                    sptr, row + 1, col + 0, c, IH, IW, C, scalar);
            ctype a11 = GetSrcData<ctype, format, bmode>::get(
                    sptr, row + 1, col + 1, c, IH, IW, C, scalar);
            /* in remap, we use float as the type of intermediate result */
            float result = static_cast<float>(a00) * (1.f - u) * (1.f - v) +
                           static_cast<float>(a01) * (1.f - u) * v +
                           static_cast<float>(a10) * (1.f - v) * u +
                           static_cast<float>(a11) * u * v;
            dst[get_offset<format>(oh, ow, c, OH, OW, C)] = round_converter(result);
        }
    }
}

template <
        typename ctype, const uint32_t format, ::BorderMode bmode,
        ::InterpolationMode imode>
void dispatch_forward(
        const ctype* src, const float* map_xy, ctype* dst, int N, int C, int IH, int IW,
        int OH, int OW, float scalar, cudaStream_t stream) {
    const int BX = 32, BY = 16;

    const int max_batch_size = 65535;
    while (N) {
        size_t curr_batch_size = N < max_batch_size ? N : max_batch_size;
        dim3 threads(BX, BY);
        dim3 blocks((OW + BX - 1) / BX, (OH + BY - 1) / BY, curr_batch_size);

        if (imode == ::InterpolationMode::INTER_NEAREST) {
            kern_general_nearest<ctype, format, bmode><<<blocks, threads, 0, stream>>>(
                    src, map_xy, dst, C, IH, IW, OH, OW, scalar);
        } else if (imode == ::InterpolationMode::INTER_LINEAR) {
            kern_general_linear<ctype, format, bmode><<<blocks, threads, 0, stream>>>(
                    src, map_xy, dst, C, IH, IW, OH, OW, scalar);
        }

        N -= curr_batch_size;
        src += curr_batch_size * C * IH * IW;
        dst += curr_batch_size * C * OH * OW;
        map_xy += curr_batch_size * OH * OW * 2;
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace remap {

template <
        typename ctype, const uint32_t format, ::BorderMode bmode,
        ::InterpolationMode imode>
void forward_proxy(
        const ctype* src, const float* map_xy, ctype* dst, int N, int C, int IH, int IW,
        int OH, int OW, float scalar, cudaStream_t stream) {
    dispatch_forward<ctype, format, bmode, imode>(
            src, map_xy, dst, N, C, IH, IW, OH, OW, scalar, stream);
    after_kernel_launch();
}

#define INST(ctype, format, bmode, imode)                                            \
    template void forward_proxy<                                                     \
            ctype, param_enumv::Remap::Format::format, ::BorderMode::bmode,          \
            ::InterpolationMode::imode>(                                             \
            const ctype*, const float*, ctype*, int, int, int, int, int, int, float, \
            cudaStream_t);

#define FOR_FORMAT_BMODE(ctype)                          \
    INST(ctype, NCHW, BORDER_CONSTANT, INTER_NEAREST)    \
    INST(ctype, NCHW, BORDER_REPLICATE, INTER_NEAREST)   \
    INST(ctype, NCHW, BORDER_REFLECT, INTER_NEAREST)     \
    INST(ctype, NCHW, BORDER_REFLECT_101, INTER_NEAREST) \
    INST(ctype, NCHW, BORDER_WRAP, INTER_NEAREST)        \
    INST(ctype, NHWC, BORDER_CONSTANT, INTER_NEAREST)    \
    INST(ctype, NHWC, BORDER_REPLICATE, INTER_NEAREST)   \
    INST(ctype, NHWC, BORDER_REFLECT, INTER_NEAREST)     \
    INST(ctype, NHWC, BORDER_REFLECT_101, INTER_NEAREST) \
    INST(ctype, NHWC, BORDER_WRAP, INTER_NEAREST)        \
    INST(ctype, NCHW, BORDER_CONSTANT, INTER_LINEAR)     \
    INST(ctype, NCHW, BORDER_REPLICATE, INTER_LINEAR)    \
    INST(ctype, NCHW, BORDER_REFLECT, INTER_LINEAR)      \
    INST(ctype, NCHW, BORDER_REFLECT_101, INTER_LINEAR)  \
    INST(ctype, NCHW, BORDER_WRAP, INTER_LINEAR)         \
    INST(ctype, NHWC, BORDER_CONSTANT, INTER_LINEAR)     \
    INST(ctype, NHWC, BORDER_REPLICATE, INTER_LINEAR)    \
    INST(ctype, NHWC, BORDER_REFLECT, INTER_LINEAR)      \
    INST(ctype, NHWC, BORDER_REFLECT_101, INTER_LINEAR)  \
    INST(ctype, NHWC, BORDER_WRAP, INTER_LINEAR)

FOR_FORMAT_BMODE(float)
DNN_INC_FLOAT16(FOR_FORMAT_BMODE(dt_float16))
DNN_INC_FLOAT16(FOR_FORMAT_BMODE(dt_bfloat16))
FOR_FORMAT_BMODE(int8_t)
FOR_FORMAT_BMODE(uint8_t)

#undef FOR_FORMAT_BMODE
#undef INST

}  // namespace remap
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
