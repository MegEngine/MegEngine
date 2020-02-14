/**
 * \file dnn/src/cuda/checksum/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./kern.cuh"

#include "src/cuda/utils.cuh"
#include "src/cuda/reduce_helper.cuh"

namespace {
    struct ChecksumOp {
        typedef uint32_t wtype;
        const uint32_t *src;
        uint32_t *dst;

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
            uint8_t* data = (uint8_t*) (src + idx);
            return (data[0] | ((uint32_t) data[1] << 8) |
                    ((uint32_t) data[2] << 16) | ((uint32_t) data[3] << 24)) *
                   (idx + 1);
        }
    };

    struct FourAlignedChecksumOp : ChecksumOp {
        __host__ __device__ uint32_t read(uint32_t idx) {
            return src[idx] * (idx + 1);
        }
    };


} // anonymous namespace

void megdnn::cuda::checksum::calc(
        uint32_t *dest,
        const uint32_t *buf,
        uint32_t *workspace,
        size_t nr_elem, cudaStream_t stream) {
    if (!nr_elem)
        return;
    if (reinterpret_cast<uint64_t>(buf) & 0b11) {
        NonFourAlignedChecksumOp op;
        op.src = buf;
        op.dst = dest;
        run_reduce<NonFourAlignedChecksumOp, false>(workspace,
                1, nr_elem, 1, stream, op);
    } else {
        FourAlignedChecksumOp op;
        op.src = buf;
        op.dst = dest;
        run_reduce<FourAlignedChecksumOp, false>(workspace,
                1, nr_elem, 1, stream, op);
    }
}

size_t megdnn::cuda::checksum::get_workspace_in_bytes(size_t nr_elem)
{
    return get_reduce_workspace_in_bytes<ChecksumOp>(1, nr_elem, 1);
}
// vim: ft=cpp syntax=cpp.doxygen
