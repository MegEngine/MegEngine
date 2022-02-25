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

template <typename ctype, const uint32_t format, ::BorderMode bmode>
struct GetSrcData {
    __device__ static inline int get_index(
            int height, int width, int channel, int h, int w, int c) {
        height = megcv::border_interpolate<bmode>(height, h);
        width = megcv::border_interpolate<bmode>(width, w);
        return get_offset<format>(height, width, channel, h, w, c);
    }
};

template <typename ctype, const uint32_t format>
struct GetSrcData<ctype, format, ::BorderMode::BORDER_CONSTANT> {
    __device__ static inline int get_index(
            int height, int width, int channel, int h, int w, int c) {
        return (height >= 0 && height < h && width >= 0 && width < w)
                     ? get_offset<format>(height, width, channel, h, w, c)
                     : -1;
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
        ctype* __restrict grad, const float* map_xy, const ctype* diff, int C, int IH,
        int IW, int OH, int OW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    grad += blockIdx.z * C * IH * IW;
    diff += blockIdx.z * C * OH * OW;
    map_xy += blockIdx.z * 2 * OH * OW;

    if (ow < OW && oh < OH) {
        float index_col = map_xy[oh * OW * 2 + ow * 2 + 0];
        float index_row = map_xy[oh * OW * 2 + ow * 2 + 1];
        int col = static_cast<int>(round_half_to_even(index_col));
        int row = static_cast<int>(round_half_to_even(index_row));
        for (int c = 0; c < C; ++c) {
            ctype hidden = diff[get_offset<format>(oh, ow, c, OH, OW, C)];
            int idx =
                    GetSrcData<ctype, format, bmode>::get_index(row, col, c, IH, IW, C);
            if (idx != -1) {
                atomic_add(grad + idx, hidden);
            }
        }
    }
}

template <typename ctype, const uint32_t format, ::BorderMode bmode>
__global__ void kern_general_linear(
        ctype* __restrict grad, const float* map_xy, const ctype* diff, int C, int IH,
        int IW, int OH, int OW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    grad += blockIdx.z * C * IH * IW;
    diff += blockIdx.z * C * OH * OW;
    map_xy += blockIdx.z * 2 * OH * OW;
    RoundingConverter<ctype> round_converter;

    if (ow < OW && oh < OH) {
        float index_col = map_xy[oh * OW * 2 + ow * 2 + 0];
        float index_row = map_xy[oh * OW * 2 + ow * 2 + 1];
        int col = static_cast<int>(floor(index_col));
        int row = static_cast<int>(floor(index_row));
        float v = index_col - col;  // alphah
        float u = index_row - row;  // alphaw
        const float one = 1.f;
        for (int c = 0; c < C; ++c) {
            float hidden =
                    static_cast<float>(diff[get_offset<format>(oh, ow, c, OH, OW, C)]);

            int a00 = GetSrcData<ctype, format, bmode>::get_index(
                    row + 0, col + 0, c, IH, IW, C);
            if (a00 != -1) {
                atomic_add(grad + a00, round_converter((one - u) * (one - v) * hidden));
            }

            int a01 = GetSrcData<ctype, format, bmode>::get_index(
                    row + 0, col + 1, c, IH, IW, C);
            if (a01 != -1) {
                atomic_add(grad + a01, round_converter((one - u) * v * hidden));
            }

            int a10 = GetSrcData<ctype, format, bmode>::get_index(
                    row + 1, col + 0, c, IH, IW, C);
            if (a10 != -1) {
                atomic_add(grad + a10, round_converter(u * (one - v) * hidden));
            }

            int a11 = GetSrcData<ctype, format, bmode>::get_index(
                    row + 1, col + 1, c, IH, IW, C);
            if (a11 != -1) {
                atomic_add(grad + a11, round_converter(u * v * hidden));
            }
        }
    }
}

template <
        typename ctype, const uint32_t format, ::BorderMode bmode,
        ::InterpolationMode imode>
void dispatch_backwarddata(
        ctype* grad, const float* map_xy, const ctype* diff, int N, int C, int IH,
        int IW, int OH, int OW, cudaStream_t stream) {
    const int BX = 32, BY = 16;
    const int max_batch_size = 65535;
    while (N) {
        size_t curr_batch_size = N < max_batch_size ? N : max_batch_size;
        dim3 threads(BX, BY);
        dim3 blocks((OW + BX - 1) / BX, (OH + BY - 1) / BY, curr_batch_size);

        cuda_check(cudaMemsetAsync(
                grad, 0, sizeof(ctype) * curr_batch_size * C * IH * IW, stream));
        if (imode == ::InterpolationMode::INTER_NEAREST) {
            kern_general_nearest<ctype, format, bmode><<<blocks, threads, 0, stream>>>(
                    grad, map_xy, diff, C, IH, IW, OH, OW);
        } else if (imode == ::InterpolationMode::INTER_LINEAR) {
            kern_general_linear<ctype, format, bmode><<<blocks, threads, 0, stream>>>(
                    grad, map_xy, diff, C, IH, IW, OH, OW);
        }

        N -= curr_batch_size;
        grad += curr_batch_size * C * IH * IW;
        diff += curr_batch_size * C * OH * OW;
        map_xy += curr_batch_size * 2 * OH * OW;
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace remap {

template <
        typename ctype, const uint32_t format, ::BorderMode bmode,
        ::InterpolationMode imode>
void backwarddata_proxy(
        ctype* grad, const float* map_xy, const ctype* diff, int N, int C, int IH,
        int IW, int OH, int OW, cudaStream_t stream) {
    dispatch_backwarddata<ctype, format, bmode, imode>(
            grad, map_xy, diff, N, C, IH, IW, OH, OW, stream);
    after_kernel_launch();
}

#define INST(ctype, format, bmode, imode)                                     \
    template void backwarddata_proxy<                                         \
            ctype, param_enumv::Remap::Format::format, ::BorderMode::bmode,   \
            ::InterpolationMode::imode>(                                      \
            ctype*, const float*, const ctype*, int, int, int, int, int, int, \
            cudaStream_t);

#define FOR_FORMAT_BMODE(ctype)                          \
    INST(ctype, NCHW, BORDER_CONSTANT, INTER_NEAREST)    \
    INST(ctype, NCHW, BORDER_REPLICATE, INTER_NEAREST)   \
    INST(ctype, NCHW, BORDER_REFLECT, INTER_NEAREST)     \
    INST(ctype, NCHW, BORDER_REFLECT_101, INTER_NEAREST) \
    INST(ctype, NCHW, BORDER_WRAP, INTER_NEAREST)        \
    INST(ctype, NCHW, BORDER_CONSTANT, INTER_LINEAR)     \
    INST(ctype, NCHW, BORDER_REPLICATE, INTER_LINEAR)    \
    INST(ctype, NCHW, BORDER_REFLECT, INTER_LINEAR)      \
    INST(ctype, NCHW, BORDER_REFLECT_101, INTER_LINEAR)  \
    INST(ctype, NCHW, BORDER_WRAP, INTER_LINEAR)

FOR_FORMAT_BMODE(float)
DNN_INC_FLOAT16(FOR_FORMAT_BMODE(dt_bfloat16))
DNN_INC_FLOAT16(FOR_FORMAT_BMODE(dt_float16))

#undef FOR_FORMAT_BMODE
#undef INST

}  // namespace remap
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
