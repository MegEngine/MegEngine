#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace warp_affine {

__device__ inline float sqr(float x) {
    return x * x;
}

__device__ inline int mod(int i, int n) {
    i %= n;
    i += (i < 0) * n;
    return i;
}

class ReplicateGetter {
public:
    __device__ int operator()(int i, int n) { return min(max(i, 0), n - 1); }
};

class ReflectGetter {
public:
    __device__ int operator()(int i, int n) {
        n <<= 1;
        i = mod(i, n);
        return min(i, n - 1 - i);
    }
};

class Reflect101Getter {
public:
    __device__ int operator()(int i, int n) {
        n = (n - 1) << 1;
        i = mod(i, n);
        return min(i, n - i);
    }
};

class WrapGetter {
public:
    __device__ int operator()(int i, int n) {
        i = mod(i, n);
        return i;
    }
};

}  // namespace warp_affine
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
