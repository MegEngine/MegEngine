#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>
#include "cuda.h"
#include "include/megdnn/dtype.h"

namespace megdnn {
namespace cuda {

#if MEGDNN_CC_CUDA
template <typename T>
static inline MEGDNN_DEVICE void atomic_add(T* address, T val);

template <>
MEGDNN_DEVICE void atomic_add<dt_float32>(dt_float32* address, dt_float32 val) {
    ::atomicAdd(reinterpret_cast<float*>(address), static_cast<float>(val));
}

// overload atomicAdd for half precision
// Taken from:
// https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomic.cuh
template <>
MEGDNN_DEVICE void atomic_add(dt_float16* address, dt_float16 val) {
#if (__CUDA_ARCH__ < 700 || __CUDACC_VER_MAJOR__ <= 9)
    unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(
            reinterpret_cast<char*>(address) - (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short data =
                reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
        dt_float16 hsum = *reinterpret_cast<dt_float16*>(&data);
        hsum += val;
        data = *reinterpret_cast<unsigned short*>(&hsum);
        old = reinterpret_cast<size_t>(address) & 2 ? (old & 0xffff) | (data << 16)
                                                    : (old & 0xffff0000) | data;
        old = ::atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
#else
    ::atomicAdd(reinterpret_cast<__half*>(address), static_cast<__half>(val));
#endif
}

template <>
MEGDNN_DEVICE void atomic_add(dt_bfloat16* address, dt_bfloat16 val) {
    unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(
            reinterpret_cast<char*>(address) - (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short data =
                reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
        dt_bfloat16 hsum = *reinterpret_cast<dt_bfloat16*>(&data);
        hsum += val;
        data = *reinterpret_cast<unsigned short*>(&hsum);
        old = reinterpret_cast<size_t>(address) & 2 ? (old & 0xffff) | (data << 16)
                                                    : (old & 0xffff0000) | data;
        old = ::atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}

template <typename T, size_t n>
struct AtomicAddIntegerImpl;

template <typename T>
struct AtomicAddIntegerImpl<T, 1> {
    inline __device__ void operator()(T* address, T val) {
        size_t offset = (size_t)address & 3;
        uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
        uint32_t old = *address_as_ui;
        uint32_t shift = offset * 8;
        uint32_t old_byte;
        uint32_t newval;
        uint32_t assumed;
        do {
            assumed = old;
            old_byte = (old >> shift) & 0xff;
            // preserve size in initial cast. Casting directly to uint32_t pads
            // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
            newval = static_cast<uint8_t>(
                    static_cast<T>(val) + static_cast<T>(old_byte));
            // newval = static_cast<uint8_t>(THCNumerics<T>::add(val,
            // old_byte));
            newval = (old & ~(0x000000ff << shift)) | (newval << shift);
            old = atomicCAS(address_as_ui, assumed, newval);
        } while (assumed != old);
    }
};

template <typename T>
struct AtomicAddIntegerImpl<T, 2> {
    inline __device__ void operator()(T* address, T val) {
        size_t offset = (size_t)address & 2;
        uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
        bool is_32_align = offset;
        uint32_t old = *address_as_ui;
        uint32_t old_bytes;
        uint32_t newval;
        uint32_t assumed;
        do {
            assumed = old;
            old_bytes = is_32_align ? old >> 16 : old & 0xffff;
            // preserve size in initial cast. Casting directly to uint32_t pads
            // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
            newval = static_cast<uint16_t>(
                    static_cast<T>(val) + static_cast<T>(old_bytes));
            // newval = static_cast<uint16_t>(THCNumerics<T>::add(val,
            // old_bytes));
            newval = is_32_align ? (old & 0xffff) | (newval << 16)
                                 : (old & 0xffff0000) | newval;
            old = atomicCAS(address_as_ui, assumed, newval);
        } while (assumed != old);
    }
};

template <>
MEGDNN_DEVICE void atomic_add(dt_int32* address, dt_int32 val) {
    ::atomicAdd(reinterpret_cast<int*>(address), static_cast<int>(val));
}

// we assume quantized int in the same tensor with same scale
template <>
MEGDNN_DEVICE void atomic_add(dt_qint32* address, dt_qint32 val) {
    ::atomicAdd(reinterpret_cast<int*>(address), val.as_int32());
}

template <>
MEGDNN_DEVICE void atomic_add(dt_int16* address, dt_int16 val) {
    AtomicAddIntegerImpl<dt_int16, sizeof(dt_int16)>()(address, val);
}

template <>
MEGDNN_DEVICE void atomic_add(dt_uint16* address, dt_uint16 val) {
    AtomicAddIntegerImpl<dt_uint16, sizeof(dt_uint16)>()(address, val);
}

// we assume quantized int in the same tensor with same scale
template <>
MEGDNN_DEVICE void atomic_add(dt_qint16* address, dt_qint16 val) {
    AtomicAddIntegerImpl<dt_int16, sizeof(dt_qint16)>()(
            reinterpret_cast<dt_int16*>(address), val.as_int16());
}
// be careful! may case over flow
#if 0
template <>
MEGDNN_DEVICE void atomic_add(dt_int8* address, dt_int8 val) {
    AtomicAddIntegerImpl<dt_int8, sizeof(dt_int8)>()(address, val);
}

template <>
MEGDNN_DEVICE void atomic_add(dt_uint8* address, dt_uint8 val) {
    AtomicAddIntegerImpl<dt_uint8, sizeof(dt_uint8)>()(address, val);
}

// we assume quantized int in the same tensor with same scale
template <>
MEGDNN_DEVICE void atomic_add(dt_quint8* address, dt_quint8 val) {
    AtomicAddIntegerImpl<dt_uint8, sizeof(dt_quint8)>()(reinterpret_cast<dt_uint8*>(address), val.as_uint8());
}

// we assume quantized int in the same tensor with same scale
template <>
MEGDNN_DEVICE void atomic_add(dt_qint8* address, dt_qint8 val) {
    AtomicAddIntegerImpl<dt_int8, sizeof(dt_qint8)>()(reinterpret_cast<dt_int8*>(address), val.as_int8());
}
#endif

#endif
}  // namespace cuda
}  // namespace megdnn