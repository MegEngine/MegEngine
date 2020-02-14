/**
 * \file dnn/src/cuda/convpooling/conv_pooling_utils.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/cuda/utils.cuh"
#include <algorithm>
#include <math.h>
#include <cuda_runtime_api.h>

//#include "./helper.cuh"


namespace megdnn {
namespace cuda {
namespace conv_pool {

#define CUDA_CHKERR(call)  \
    do { \
        cudaError_t code = (call); \
            megdnn_assert(code == cudaSuccess, "cuda err %d: %s (call %s at %s:%s:%d)", \
                    int(code), cudaGetErrorString(code), # call, \
                        __FILE__, __func__, __LINE__); \
    } while(0)

#define CUDA_CHK_KERN_ERR CUDA_CHKERR(cudaDeviceSynchronize());

static inline int __host__ align_to_warp(int n) {
    int x = n / 32 * 32;
    if (!x)
        x = n;
    return x;
}

// --- Nonline ---
struct Relu {
    static __device__ float apply(float x) {
        return x > 0 ? x : 0;
    }
};

struct Sigmoid {
    static __device__ float apply(float x) {
        float exp_value = exp((double) -x);
        return 1 / (1 + exp_value);
    }
};

struct Identity {
    static __device__ float apply(float x) {
        return x;
    }
};

// --- Static Reduce ---
template<int size, class Op>
struct StaticReduce {
    static __device__ float apply(const float *val) {
        const int half = size / 2;
        return Op::apply(
                StaticReduce<half, Op>::apply(val),
                StaticReduce<size - half, Op>::apply(val + half));
    }
};

template<class Op>
struct StaticReduce<1, Op> {
    static __device__ float apply(const float *val) {
        return val[0];
    }
};

template<class Op>
struct StaticReduce<2, Op> {
    static __device__ float apply(const float *val) {
        return Op::apply(val[0], val[1]);
    }
};

struct OpAdd {
    static __device__ float apply(float a, float b) {
        return a + b;
    }
};

struct OpMax {
    static __device__ float apply(float a, float b) {
        return max(a, b);
    }
};

struct IdxGetterConvolution {
    static inline __device__ int apply(int kern, int i, int p) {
        return kern - i - 1 + p;
    }

};

struct IdxGetterCorrRel {
    static inline __device__ int apply(int kern, int i, int p) {
        return i - p;
    }
};


// --- Pooling ---
struct MeanPooler {
    template<int pool_shape_h, int pool_shape_w>
    static __device__ float apply(const float *val) {
        const int size = pool_shape_h * pool_shape_w;
        return StaticReduce<size, OpAdd>::apply(val) / size;
    }
};

struct MaxPooler {
    template<int pool_shape_h, int pool_shape_w>
    static __device__ float apply(const float *val) {
        return StaticReduce<pool_shape_h * pool_shape_w, OpMax>::apply(val);
    }
};


    // --- Reader ---
class Tex1DReader {
    cudaTextureObject_t m_tex;
    int m_base_offset, m_chl_stride, m_row_stride, m_row_offset;
    //size_t batch_, chal_, height_, weight_;

    public:
        // Set attributes of texture Object
        /*__device__ void init(cudaTextureObject_t& tex,
            size_t batch, size_t chal, size_t height, size_t weight) {
            batch_  = batch;
            chal_   = chal;
            height_ = height;
            weight_ = weight;
            m_chl_stride  = height * weight;
            m_row_stride  = weight;
        }

        __device__ void set_pos(cudaTextureObject_t& tex,
            // Current position
            size_t n, size_t c, size_t h, size_t w) {
            m_tex   = tex;
            m_base_offset = ((n * chal_ + c) * height_ + h) * weight_ + w;
        }
        */
        __device__ void set_pos(cudaTextureObject_t& tex,
            // Current position
            int chal, int height, int weight, int n, int c, int h, int w) {
            m_chl_stride  = height * weight;
            m_row_stride  = weight;
            m_tex   = tex;
            m_base_offset = ((n * chal + c) * height + h) * weight + w;
        }

        __device__ void reset_row() {
            m_row_offset = m_base_offset;
        }

        __device__ void next_row() {
            m_row_offset += m_row_stride;
        }

        __device__ void next_channel() {
            m_base_offset += m_chl_stride;
        }

        __device__ float get(int /*dr*/, int dc) {
            return tex1Dfetch<float>(m_tex, dc + m_row_offset);
        }

        __device__ float get(int idx) {
            return tex1Dfetch<float>(m_tex, idx + m_base_offset);
        }
};

 extern __host__  void create_cuda_tex(float *input, cudaTextureObject_t& tex,
        size_t N, size_t IC, size_t IH, size_t IW);



} // namespace conv_pool
} // namespace cuda
} // namespace megdnn
// vim: syntax=cpp.doxygen
