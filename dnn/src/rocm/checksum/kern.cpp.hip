/**
 * \file src/rocm/checksum/kern.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "hip_header.h"
#include "./kern.h.hip"

#include "src/rocm/reduce_helper.h.hip"

namespace megdnn {
namespace rocm {
namespace checksum {

namespace {
struct ChecksumOp {
    typedef uint32_t wtype;
    const uint32_t* src;
    uint32_t* dst;

    static const uint32_t INIT = 0;

    __host__ __device__ void write(uint32_t idx, uint32_t val) {
        dst[idx] = val;
    }

    __host__ __device__ static uint32_t apply(uint32_t a, uint32_t b) {
        return a + b;
    }
};

struct NonFourAlignedChecksumOp : ChecksumOp {
    __host__ __device__ uint32_t read(uint32_t idx) {
        uint8_t* data = (uint8_t*)(src + idx);
        return (data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16) |
                ((uint32_t)data[3] << 24)) *
               (idx + 1);
    }
};

struct FourAlignedChecksumOp : ChecksumOp {
    __host__ __device__ uint32_t read(uint32_t idx) {
        return src[idx] * (idx + 1);
    }
};

}  // anonymous namespace

void calc(uint32_t* dest, const uint32_t* buf, uint32_t* workspace,
          size_t nr_elem, hipStream_t stream) {
    if (!nr_elem)
        return;
    if (reinterpret_cast<uint64_t>(buf) & 0b11) {
        NonFourAlignedChecksumOp op;
        op.src = buf;
        op.dst = dest;
        run_reduce<NonFourAlignedChecksumOp, false>(workspace, 1, nr_elem, 1,
                                                    stream, op);
    } else {
        FourAlignedChecksumOp op;
        op.src = buf;
        op.dst = dest;
        run_reduce<FourAlignedChecksumOp, false>(workspace, 1, nr_elem, 1,
                                                 stream, op);
    }
}

size_t get_workspace_in_bytes(size_t nr_elem) {
    return get_reduce_workspace_in_bytes<ChecksumOp>(1, nr_elem, 1);
}

}  // namespace checksum
}  // namespace rocm`
}  // namespace megdnn


// vim: ft=cpp syntax=cpp.doxygen
