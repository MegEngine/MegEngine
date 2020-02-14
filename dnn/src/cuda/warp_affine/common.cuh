/**
 * \file dnn/src/cuda/warp_affine/common.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

namespace megdnn {
namespace cuda {
namespace warp_affine {

__device__ inline float sqr(float x)
{
    return x * x;
}

__device__ inline int mod(int i, int n)
{
    i %= n;
    i += (i < 0) * n;
    return i;
}

class ReplicateGetter {
    public:
        __device__ int operator()(int i, int n)
        {
            return min(max(i, 0), n-1);
        }
};

class ReflectGetter {
    public:
        __device__ int operator()(int i, int n)
        {
            n <<= 1;
            i = mod(i, n);
            return min(i, n-1-i);
        }
};

class Reflect101Getter {
    public:
        __device__ int operator()(int i, int n)
        {
            n = (n-1)<<1;
            i = mod(i, n);
            return min(i, n-i);
        }
};

class WrapGetter {
    public:
        __device__ int operator()(int i, int n)
        {
            i = mod(i, n);
            return i;
        }
};

} // namespace warp_affine
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen

